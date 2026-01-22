import inspect
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from atom.config import CUDAGraphMode, Config, get_current_atom_config
from atom.utils import envs
from atom.utils.cuda_graph import CUDAGraphWrapper
from atom.utils.dbo.utils import current_platform, set_graph_pool_id
import torch

from atom.utils.dbo.ubatching import UBatchContext, make_ubatch_contexts
from atom.utils.forward_context import (
    DPMetadata,
    create_forward_context,
    get_forward_context,
    override_forward_context,
)
from aiter import logger
from aiter.dist.parallel_state import get_ep_group

@dataclass
class UbatchMetadata:
    context: UBatchContext
    input_ids: torch.Tensor
    positions: torch.Tensor
    num_tokens: int

@dataclass
class CUDAGraphMetaData:
    cudagraph: torch.cuda.CUDAGraph
    ubatch_metadata: UbatchMetadata
    outputs: Any | None = None


class UBatchWrapper:
    def __init__(
        self,
        runnable: Callable,
        atom_config: Config,
        runtime_mode: CUDAGraphMode,
        device: torch.cuda.device,
    ):
        self.runnable = runnable
        self.atom_config = atom_config
        self.compilation_config = atom_config.compilation_config
        self.comm_stream = torch.cuda.Stream(device=device)
        # Two ubatch threads plus the main thread
        self.ready_barrier = threading.Barrier(3)

        self.cudagraphs: dict[int, CUDAGraphMetaData] = {}

        self.cudagraph_wrapper = None
        self.graph_pool = None
        if runtime_mode is not CUDAGraphMode.NONE:
            self.cudagraph_wrapper = CUDAGraphWrapper(
                runnable, atom_config, runtime_mode=runtime_mode
            )
            self.graph_pool = current_platform.get_global_graph_pool()

        self.device = device


    def __getattr__(self, key: str):
        # allow accessing the attributes of the runnable.
        if hasattr(self.runnable, key):
            return getattr(self.runnable, key)
        raise AttributeError(
            f"Attribute {key} not exists in the runnable of "
            f"cudagraph wrapper: {self.runnable}"
        )

    def unwrap(self) -> Callable:
        # in case we need to access the original runnable.
        return self.runnable

    def _capture_ubatches(self, ubatch_metadata, model) -> torch.Tensor:
        """
        Capture a cudagraph for a microbatched run.

        The logic here is somewhat complicated because we need to make sure that
        each of the ubatch threads initialize the cuda context before we start
        the graph capture.

        The flow is as follows:
        1. The main thread starts up each ubatch thread. Each thread will
        initialize its cuda context (torch.cuda.current_blas_handle())
        before going to sleep upon entering the ubatch_context.

        2. The main thread starts the graph capture and wakes up the first
        ubatch thread.

        3. Each ubatch thread runs the model to completion and returns the
        completed output tensors back to the main thread.

        4. The main thread stores the captured cudagraph along with its metadata
        and returns
        """

        @torch.inference_mode()
        def _capture_ubatch_thread(results, ubatch_metadata):
            torch.cuda.set_device(self.device)
            ubatch_context = ubatch_metadata.context
            with torch.cuda.stream(ubatch_context.compute_stream):
                _ = torch.cuda.current_blas_handle()
            with torch.cuda.stream(ubatch_context.comm_stream):
                _ = torch.cuda.current_blas_handle()
            with ubatch_context:
                print('This is capture_ubatches', ubatch_metadata.input_ids.shape, ubatch_metadata.positions.shape)
                model_output = model(
                    input_ids=ubatch_metadata.input_ids,
                    positions=ubatch_metadata.positions,
                    intermediate_tensors=None,
                )

            results.append((ubatch_metadata.context.id, model_output))

        results: list[tuple[int, torch.Tensor]] = []
        compute_stream = ubatch_metadata[0].context.compute_stream
        num_tokens = ubatch_metadata[0].num_tokens + ubatch_metadata[1].num_tokens

        # Ubatches will manually manage the forward context, so we override
        # it to None here so we can have it restored correctly later
        with override_forward_context(None):
            ubatch_threads = []
            for metadata in ubatch_metadata:
                thread = threading.Thread(
                    target=_capture_ubatch_thread,
                    args=(
                        results,
                        metadata,
                    ),
                )
                ubatch_threads.append(thread)
                thread.start()
            self.ready_barrier.wait()  # Wait for both threads to be ready

            # Capture the cudagraph
            cudagraph_metadata = CUDAGraphMetaData(
                cudagraph=torch.cuda.CUDAGraph(),
                ubatch_metadata=ubatch_metadata,
            )
            if self.graph_pool is not None:
                set_graph_pool_id(self.graph_pool)
            else:
                set_graph_pool_id(current_platform.graph_pool_handle())
            with torch.cuda.graph(
                cudagraph_metadata.cudagraph,
                stream=compute_stream,
                pool=self.graph_pool,
            ):
                ubatch_metadata[0].context.cpu_wait_event.set()
                for thread in ubatch_threads:
                    thread.join()
                sorted_results = [value for position, value in sorted(results)]
                result = torch.cat(sorted_results, dim=0)
                cudagraph_metadata.outputs = result
            self.cudagraphs[num_tokens] = cudagraph_metadata
        return cudagraph_metadata.outputs

    def _run_ubatches(self, ubatch_metadata, model) -> torch.Tensor:
        thread_exceptions = []

        @torch.inference_mode()
        def _ubatch_thread(results, model, ubatch_metadata):
            print("This is run_ubatches")
            try:
                with ubatch_metadata.context:
                    forward_method = getattr(model, 'forward', model.__call__)
                    sig = inspect.signature(forward_method)
                    kwargs = {
                        'input_ids': ubatch_metadata.input_ids,
                        'positions': ubatch_metadata.positions,
                    }
                    if 'intermediate_tensors' in sig.parameters:
                        kwargs['intermediate_tensors'] = None
                    model_output = model(**kwargs)
                results.append((ubatch_metadata.context.id, model_output))
            except Exception as e:
                import traceback
                thread_exceptions.append((ubatch_metadata.context.id, e, traceback.format_exc()))

        results: list[tuple[int, torch.Tensor]] = []

        # Ubatch threads will manually manage the forward context, so we
        # override it to None here so we can have it restored correctly
        # after both threads have finished
        with override_forward_context(None):
            ubatch_threads = []
            for metadata in ubatch_metadata:
                thread = threading.Thread(
                    target=_ubatch_thread,
                    args=(
                        results,
                        model,
                        metadata,
                    ),
                )
                ubatch_threads.append(thread)
                thread.start()
            self.ready_barrier.wait()  # Wait for both threads to be ready
            ubatch_metadata[0].context.cpu_wait_event.set()
            for thread in ubatch_threads:
                thread.join()

        # Check if any thread raised an exception
        if thread_exceptions:
            for thread_id, exc, tb in thread_exceptions:
                logger.error(f"Ubatch thread {thread_id} failed with exception:\n{tb}")
            # Re-raise the first exception
            raise thread_exceptions[0][1]

        sorted_results = [value for position, value in sorted(results)]
        result = torch.cat(sorted_results, dim=0)
        return result

    def _make_ubatch_metadata(
        self,
        ubatch_slices,
        attn_metadata,
        input_ids,
        positions,
        compute_stream,
        # dp_metadata,
        context,
    ) -> list[UbatchMetadata]:
        # Create one forward context per ubatch
        forward_contexts = []
        for i, ubatch_slice in enumerate(ubatch_slices):
            forward_contexts.append(
                create_forward_context(
                    attn_metadata[i] if attn_metadata is not None else None,
                    self.atom_config,
                    context=context,
                    # dp_metadata=dp_metadata,
                    ubatch_slices=ubatch_slices,
                )
            )

        ubatch_ctxs = make_ubatch_contexts(
            num_micro_batches=len(ubatch_slices),
            comm_stream=self.comm_stream,
            compute_stream=compute_stream,
            forward_contexts=forward_contexts,
            ready_barrier=self.ready_barrier,
        )

        ubatch_metadata: list[UbatchMetadata] = []
        for i, ubatch_slice in enumerate(ubatch_slices):
            (
                sliced_input_ids,
                sliced_positions,
            ) = self._slice_model_inputs(
                ubatch_slice.token_slice,
                input_ids,
                positions,
            )
            ubatch_metadata.append(
                UbatchMetadata(
                    context=ubatch_ctxs[i],
                    input_ids=sliced_input_ids,
                    positions=sliced_positions,
                    num_tokens=ubatch_slice.token_slice.stop
                    - ubatch_slice.token_slice.start,
                )
            )

        return ubatch_metadata

    def _slice_model_inputs(
        self,
        tokens_slice: slice,
        input_ids,
        positions,
    ):
        sliced_input_ids = input_ids[tokens_slice]
        # if we are using mrope. Mrope adds an additional dimension to the
        # positions tensor
        if positions.ndim == 2:
            sliced_positions = positions[:, tokens_slice]
        else:
            sliced_positions = positions[tokens_slice]


        return (
            sliced_input_ids,
            sliced_positions,
        )

    def __call__(self, *args, **kwargs):
        logger.info(' Calling UBatchWrapper for DBO...')
        forward_context = get_forward_context()
        ubatch_slices = forward_context.ubatch_slices
        config = get_current_atom_config()
        runtime_mode = config.compilation_config.cudagraph_mode
        cudagraph_runtime_mode = runtime_mode

        # If there's no ubatching, just run the runnable object
        if ubatch_slices is None:
            # This is to account for the case where ubatching was aborted.
            # When we capture full graphs we only capture one graph per shape,
            # meaning that if we have a ubatched  cudagraph for the current
            # num_tokens, we don't have a non-ubatched one. Without this
            # check, the cudagraph wrapper will try to capture a cudagraph
            # for this shape during a normal run.
            # if cudagraph_runtime_mode is CUDAGraphMode.FULL:
            #     assert batch_descriptor is not None
            #     if batch_descriptor.num_tokens in self.cudagraphs:
            #         cudagraph_runtime_mode = CUDAGraphMode.NONE

            if cudagraph_runtime_mode in (CUDAGraphMode.NONE, CUDAGraphMode.PIECEWISE):
                return self.runnable(*args, **kwargs)
            else:
                assert self.cudagraph_wrapper is not None
                return self.cudagraph_wrapper(*args, **kwargs)

        attn_metadata = forward_context.attn_metadata
        num_tokens = (
            ubatch_slices[0].token_slice.stop - ubatch_slices[0].token_slice.start
        ) * 2
        input_ids = args[0]
        positions = args[1]
        compute_stream = torch.cuda.current_stream()

        # dp_metadata = forward_context.dp_metadata

        # We shouldn't be here unless we are running with multiple DP ranks
        # assert dp_metadata is not None
        num_tokens_per_ubatch = (
            ubatch_slices[0].token_slice.stop - ubatch_slices[0].token_slice.start
        )
        dp_size = self.atom_config.parallel_config.data_parallel_size
        # ubatch_num_tokens_across_dp = torch.tensor(
        #     [num_tokens_per_ubatch] * dp_size, device="cpu", dtype=torch.int32
        # )
        # ubatch_dp_metadata = DPMetadata.make(
        #     self.atom_config.parallel_config,
        #     num_tokens_per_ubatch,
        #     ubatch_num_tokens_across_dp,
        # )

        if (
            num_tokens not in self.cudagraphs
            and cudagraph_runtime_mode is CUDAGraphMode.FULL
        ):
            ubatch_metadata = self._make_ubatch_metadata(
                ubatch_slices=ubatch_slices,
                attn_metadata=attn_metadata,
                input_ids=input_ids,
                positions=positions,
                compute_stream=compute_stream,
                # dp_metadata=ubatch_dp_metadata,
                context=forward_context.context,
            )

            return self._capture_ubatches(ubatch_metadata, self.model)
        elif (
            num_tokens in self.cudagraphs
            and cudagraph_runtime_mode is CUDAGraphMode.FULL
        ):
            cudagraph_metadata = self.cudagraphs[num_tokens]
            cudagraph_metadata.cudagraph.replay()
            return cudagraph_metadata.outputs
        else:
            ubatch_metadata = self._make_ubatch_metadata(
                ubatch_slices=ubatch_slices,
                attn_metadata=attn_metadata,
                input_ids=input_ids,
                positions=positions,
                compute_stream=compute_stream,
                # dp_metadata=dp_metadata,
                context=forward_context.context,
            )
            return self._run_ubatches(ubatch_metadata, self.model)


