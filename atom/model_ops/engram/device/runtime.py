# SPDX-License-Identifier: MIT
"""Engram's device-side runtime: the UVA lookup, and the inputs a forward gets.

`EngramUva` is the half of `host.EngramHost` that reaches Triton -- hashing
where the token ids already are, and the gather that reads a page-locked shard.
The host builds one only after it has registered that shard, which is why
nothing here re-decides whether UVA is on.
"""

from contextlib import ExitStack
from dataclasses import dataclass

import numpy as np
import torch

from atom.model_ops.engram.device.hashing import EngramHashTables, engram_row_indices
from atom.model_ops.engram.device.staging import EngramStaging
from atom.model_ops.engram.device.uva import uva_gather_into
from atom.model_ops.engram.host import EngramHost, EngramPrefetcher, EngramRequest
from atom.utils import CpuGpuBuffer, envs


class EngramUva:
    """Hash on the device, read the rows over the bus, reassemble across TP.

    Only the row INDICES reach the gather: the kernel reads the table rows out
    of page-locked host memory and dequantizes them there, so neither the gather
    nor the fp8 decode runs on the host and there is no embedding H2D.

    Each rank owns a slice of the hash heads and writes zeros for the rest, so
    the all-gather that reassembles the full width is a concatenation; the
    projection that consumes it is replicated.
    """

    def __init__(self, host, num_hash_heads):
        self.host = host
        self.hash_tables = EngramHashTables.from_mapping(
            host.prefetcher._hash_mapping, host.device
        )
        # One buffer for every layer: a layer's rows are consumed by its own
        # gather before the next layer is hashed.
        self.row_ids = torch.empty(
            host.max_num_tokens, num_hash_heads, dtype=torch.int64, device=host.device
        )
        self._ids_staging = None
        self.overlap = EngramStaging(self) if envs.ATOM_ENGRAM_OVERLAP else None

    def _rows_from_host(self, requests, rows):
        """Hash on the host and stage the indices through pinned memory.

        `from_numpy(...).to(device)` would copy from PAGEABLE memory, where
        `non_blocking` is ignored and the driver stages through its own bounce
        buffer every step.
        """
        host = self.host
        if self._ids_staging is None:
            self._ids_staging = CpuGpuBuffer(
                host.max_num_tokens,
                host.total_heads,
                dtype=torch.int64,
                device=host.device,
                pin_memory=True,
            )
        per_layer = host.prefetcher.row_indices(requests)

        def upload(layer):
            self._ids_staging.np[:rows] = per_layer[layer]
            return self._ids_staging.copy_to_gpu(rows)

        return upload

    def _rows_from_device(self, batch, rows):
        """Hash where the ids are, from the very tensor the model embeds."""

        def hashed(layer):
            return engram_row_indices(
                self.hash_tables,
                layer,
                batch.compressed,
                batch.batch_ids,
                batch.cu_seqlens,
                batch.history,
                batch.history_index,
                self.row_ids[:rows],
            )

        return hashed

    def stage(self, requests, rows, staged, batch=None):
        """Fill the host's device buffers for `rows`, zeroing the padding."""
        host = self.host
        indices = (
            self._rows_from_device(batch, rows)
            if batch is not None
            else self._rows_from_host(requests, rows)
        )
        for layer, buffer in host.buffers.items():
            ids = indices(layer)
            table = host.prefetcher._tables[layer]
            if ids.shape != (rows, host.total_heads):
                raise RuntimeError(
                    f"engram UVA indices are {tuple(ids.shape)}, expected "
                    f"{(rows, host.total_heads)}"
                )
            # empty, not zeros: the kernel stores every row it is given, writing
            # zeros itself for the heads this rank does not own. Flat
            # `[tokens, local_heads * head_dim]`, which is the same bytes the
            # kernel writes and the layout the all-gather below wants.
            flat = torch.empty(
                rows,
                host.local_heads * table.head_dim,
                dtype=buffer.gpu.dtype,
                device=host.device,
            )
            uva_gather_into(
                table,
                ids,
                flat.view(rows, host.local_heads, table.head_dim),
                head_start=host.head_start,
                local_heads=host.local_heads,
                total_heads=host.total_heads,
            )
            out = flat
            if host._tp_group is not None:
                # Gather on the LAST dim of a 2-D view, which is what routes this
                # through aiter's IPC all-gather instead of NCCL: the custom path
                # needs dim 0 or a 16-byte-aligned last dim, and a head slice is
                # `local_heads * head_dim * 2` bytes wide. NCCL is not just
                # slower here -- its end event, recorded during a CUDAGraph
                # capture, is later read by the watchdog and crashes with
                # hipErrorCapturedEvent (see moe.all_gather_with_padding).
                # Rank-major concatenation puts head `h` back at column
                # `h * head_dim`, so the result needs no transpose; the trailing
                # columns are the padding an indivisible head count leaves.
                out = host._tp_group.all_gather(out, use_custom=True, dim=1)
                out = out[:, : host.embed_width]
            buffer.gpu[:rows].copy_(out)
            if staged > rows:
                buffer.gpu[rows:staged].zero_()
        return host.mark_staged(staged)


@dataclass(frozen=True)
class EngramBatch:
    """One forward as the kernels see it: every field already on device.

    `compressed` is derived from the very tensor the model embeds, so the rows
    Engram looks up and the tokens the model runs cannot disagree. `history` is
    any `[n, max_ngram_size - 1]` int64 plane with `history_index` naming a row
    per request, which is how the committed cursor is read where it lies rather
    than gathered out first.
    """

    compressed: torch.Tensor
    batch_ids: torch.Tensor
    cu_seqlens: torch.Tensor
    history: torch.Tensor
    history_index: torch.Tensor


@dataclass(frozen=True)
class EngramInputs:
    embeddings: dict[int, torch.Tensor]
    histories: np.ndarray
    compressed_rows: tuple[np.ndarray, ...]


class EngramInputPreparer:
    """Prepare finalized runtime tokens using the cache's restored history.

    No request-history dictionary: the returned history commits with the model
    state, and generic STATE checkpoints carry it across migration and reuse.
    """

    def __init__(self, mapping, host, resources=None):
        self.mapping, self.host, self.resources = mapping, host, resources

    @classmethod
    def from_checkpoint(cls, directory, config, max_tokens, device):
        from transformers import AutoTokenizer

        from atom.model_loader.deepseek_v41 import engram_tables
        from atom.model_ops.engram.mapping import (
            CompressedTokenizer,
            EngramConfig,
            NgramHashMapping,
        )

        resources = ExitStack()
        try:
            tables = resources.enter_context(engram_tables(directory, config))
            tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True)
            engram_config = EngramConfig.from_hf(config.to_dict())
            mapping = NgramHashMapping(
                engram_config,
                CompressedTokenizer(
                    tokenizer, expected_size=engram_config.compressed_vocab_size
                ),
            )
            host = EngramHost(
                EngramPrefetcher(mapping, tables),
                max_tokens,
                engram_config.num_hash_heads,
                engram_config.head_dim,
                device,
                device_lookup=EngramUva,
            )
            resources.callback(host.shutdown)
            return cls(mapping, host, resources)
        except BaseException:
            resources.close()
            raise

    def prepare(
        self,
        spans,
        token_ids,
        histories,
        *,
        dummy=False,
        token_mask=None,
        padded_rows=None,
        batch=None,
        cursor_positions=None,
        cursor_out=None,
    ):
        """Stage one embedding row per row the forward will run.

        `token_ids` are the rows the requests own. `padded_rows` is the width
        the forward runs, which is wider whenever the batch was padded up to a
        captured shape -- the tail belongs to no request, so it is staged as
        zeros rather than looked up, and it cannot be read off `token_ids`
        because the padding is applied to the model's input after this.

        `batch` moves the n-gram hashing to the device (`EngramBatch`). The
        readback below stays: `compressed_rows` and the advanced history are
        still worked out here, and both want the ids on the host. What goes is
        the per-layer, per-request hashing -- measured at 4.4 ms of every 48 ms
        decode step, with the device idle for all of it.
        """
        compressed_rows = []
        rows = token_ids.numel() if padded_rows is None else padded_rows
        if self.host.overlap is not None and (batch is not None or dummy):
            return EngramInputs(
                self.host.overlap.prepare(
                    batch,
                    rows,
                    cursor_positions=cursor_positions,
                    cursor_out=cursor_out,
                ),
                histories,
                (),
            )
        if dummy:
            self.host.stage_dummy(rows)
            next_histories = histories
        elif batch is not None:
            # Nothing left for the host to read the ids for: the rows are
            # hashed from them on the device and the advanced history is
            # written there too, so the readback below goes with the work it
            # was feeding. `histories` passes through because the caller still
            # holds the committed one; the next one now lives in the cursor.
            self.host.stage_embeddings((), padded_rows=rows, batch=batch)
            next_histories = histories
        else:
            # These are the final GPU IDs, including deferred decode tokens.
            # One D2H per batch is the eager host-lookup contract; the device
            # path above is the one that retires it.
            ids = token_ids.detach().cpu().numpy()
            requests, next_histories = [], []
            for span, history in zip(spans, histories):
                tokens = ids[span.token_slice]
                mask = None if token_mask is None else token_mask[span.token_slice]
                requests.append(
                    EngramRequest(
                        span.request_id,
                        0,
                        span.position,
                        tuple(tokens),
                        tuple(history),
                        token_mask=None if mask is None else tuple(mask),
                    )
                )
                compressed = self.mapping.compress_tokens(
                    tokens[None, :], None if mask is None else mask[None, :]
                )
                compressed_rows.append(compressed[0])
                next_histories.append(
                    self.mapping.advance_history(history[None, :], compressed)[0]
                )
            self.host.stage_embeddings(requests, padded_rows=rows)
            next_histories = np.asarray(next_histories, dtype=np.int64).reshape(
                histories.shape
            )
        self.host.wait_for_embeddings()
        return EngramInputs(
            {
                layer: self.host.embeddings(layer).unsqueeze(0)
                for layer in self.host.layer_ids
            },
            next_histories,
            tuple(compressed_rows),
        )

    def close(self):
        if self.resources is not None:
            self.resources.close()
            self.resources = None
        else:
            self.host.shutdown()
