"""Narrow 81198 homogeneous HMA bridge; no source edits and no wire changes.

runtime_install(connector) arms one owner per process before warmup. Metadata
bind registers final plans; geometry/session attachment waits for the ORIGINAL
lazy _get_built_session result, never an extra build/handshake. Dynamic sidecar
uses EXISTING multiproc MessageQueue codec; Ray/other transports are unverified.
After warmup admission stops, scheduler stop_sampling/close_scheduler precede
serialized worker close_worker RPCs. No call drains/polls; NOT_READY blocks the
formal window. The harness MUST aggregate all ranks and supply quiescence.
"""
import copy
from collections import Counter


class Adapter:
    def __init__(self, observer):
        self.observer = observer
        self.pending = {}
        self.restores = []
        self.worker = None
        self.scheduler = None
        self.deferred = {}
        self.sampling = True
        self.exported = []
        self.sample_limit = 32
        self.sampled = 0
        self.selection = []
        self.bound = set()
        self.capacity = None

    def _replace(self, obj, name, fn):
        self.restores.append((obj, name, name in vars(obj), vars(obj).get(name)))
        setattr(obj, name, fn)

    def install_scheduler(self, scheduler):
        if not self.observer.enabled:
            return
        self.scheduler = scheduler
        update = scheduler.update_state_after_alloc
        build = scheduler.build_connector_meta

        def observed_update(request, blocks, num_external_tokens, connector_worker=None):
            if not self.observer.enabled or not self.sampling:
                return update(request, blocks, num_external_tokens, connector_worker)
            # Capture authoritative P/full D BEFORE product mutates/clears params.
            snapshot = self.observer._safe(lambda: (
                copy.deepcopy(blocks.get_block_ids()),
                copy.deepcopy(request.kv_transfer_params)))
            result = update(request, blocks, num_external_tokens, connector_worker)
            if snapshot is not None:
                self.observer._safe(self._capture, scheduler, request,
                                    num_external_tokens, snapshot)
            return result

        def observed_build(output):
            if not self.observer.enabled:
                return build(output)
            sidecar = self.observer._safe(lambda: {
                req: copy.deepcopy(self.pending[req])
                for req in scheduler._reqs_need_recv if req in self.pending})
            result = build(output)
            if sidecar is not None:
                self.observer._safe(setattr, result, 'prefix_read_observer', sidecar)
                self.exported.extend(sidecar.values())
                for req in sidecar:
                    self.pending.pop(req, None)
            return result

        self._replace(scheduler, 'update_state_after_alloc', observed_update)
        self._replace(scheduler, 'build_connector_meta', observed_build)

    def _capture(self, scheduler, request, external, snapshot):
        if not scheduler._has_mamba or len(scheduler._attn_group_ids) != 1:
            raise ValueError('only homogeneous attention+KDA layout supported')
        if request.request_id not in scheduler._reqs_need_recv:
            return
        raw, producer_params = snapshot
        if request.request_id in self.pending:
            raise ValueError('repeated allocation before metadata build')
        attention, state = scheduler.split_block_groups(raw)
        producer = producer_params['remote_block_ids']
        if len(state) != len(producer) - 1 or any(len(g) != 1 for g in state):
            raise ValueError('unsupported KDA running-state alignment')
        if any(len(g) != 1 for g in producer[1:]):
            raise ValueError('unsupported producer KDA alignment')
        groups = scheduler.kv_cache_config.transfer_groups
        group_ids = [*scheduler._attn_group_ids, *scheduler._mamba_group_ids]
        spec = groups[group_ids[0]].kv_cache_spec
        parallel = scheduler.vllm_config.parallel_config
        span = spec.block_size * (parallel.decode_context_parallel_size
                                  if spec.dcp_sharded else 1)
        params = scheduler._req_kv_params[request.request_id]
        transfer = params['transfer_id']
        eligible = external > 0 and request.num_prompt_tokens - 1 - external >= span
        reason = 'eligible' if eligible else 'no_complete_cached_page_with_external'
        if eligible and self.sampled >= self.sample_limit:
            reason = 'sample_limit'
        sampled = reason == 'eligible'
        self.sampled += int(sampled)
        self.pending[request.request_id] = dict(sampled=sampled, reason=reason,
            request=request.request_id, transfer=transfer, episode=transfer,
            prompt=request.num_prompt_tokens, external=external,
            span=[request.num_prompt_tokens - 1 - external,
                  request.num_prompt_tokens - 1], page_tokens=span,
            raw_full_local=[list(g) for g in raw], full_local=[attention, *state],
            producer=copy.deepcopy(producer),
            groups=[list(groups[i].layer_names) for i in group_ids])

    def install_worker(self, worker):
        if not self.observer.enabled:
            return
        self.worker = worker
        self.observer.install(worker)
        lookup = worker._get_built_session
        def observed_lookup(engine):
            if not self.observer.enabled or self.observer.bypass.get():
                return lookup(engine)
            result = lookup(engine)
            self.observer._safe(self._attach_sessions, engine, result)
            return result
        self._replace(worker, '_get_built_session', observed_lookup)

    def bind(self, metadata):
        """At worker bind boundary; errors invalidate evidence, not serving."""
        if not self.observer.enabled:
            return
        self.observer._safe(self._bind, metadata)

    def _bind(self, metadata):
        worker = self.worker
        sidecar = metadata.prefix_read_observer
        if set(sidecar) != set(metadata.reqs_to_recv):
            raise ValueError('sidecar/request inventory mismatch')
        for req, meta in metadata.reqs_to_recv.items():
            p = copy.deepcopy(sidecar[req])
            if p['transfer'] != meta.transfer_id or p['request'] != req:
                raise ValueError('sidecar identity mismatch')
            if meta.tp_size != worker.world_size or meta.remote_dp_rank != 0:
                raise ValueError('only homogeneous TP/single producer DP supported')
            key = (req, meta.transfer_id, worker.tp_rank)
            if key in self.bound:
                raise ValueError('repeated allocation')
            self.bound.add(key)
            if type(p['sampled']) is not bool or p['reason'] not in ('eligible', 'no_complete_cached_page_with_external', 'sample_limit'):
                raise ValueError('missing/invalid sampling classification')
            self.selection.append(dict(request=req, transfer=meta.transfer_id,
                                       sampled=p['sampled'], reason=p['reason']))
            if not p['sampled']:
                if p['reason'] == 'eligible':
                    raise ValueError('inconsistent skip classification')
                self.observer.skipped.add(key)
                continue
            if p['reason'] != 'eligible':
                raise ValueError('inconsistent sampled classification')
            p.update(rank=worker.tp_rank)
            self.observer.register(p, {})
            self.deferred[key] = (p, meta)

    def _attach_sessions(self, engine, result):
        from vllm.distributed.kv_transfer.kv_connector.v1.moriio import moriio_layout as layout
        context = self.observer.context.get()
        if context is None:
            raise ValueError('session lookup without request context')
        key = context[0]
        p, meta = self.deferred.pop(key)
        worker = self.worker
        expected_engine = worker.get_engine_name_with_dp(meta.remote_engine_id, meta.remote_dp_rank)
        if engine != expected_engine:
            raise ValueError('late session peer mismatch')
        rank = worker._remote_tp_rank(meta.tp_size)
        sessions, remote = result
        p.update(layout='homogeneous-strided-v1', role='D', rank=worker.tp_rank,
                 ranks=list(range(worker.world_size)),
                 peer=dict(engine=meta.remote_engine_id, tp_rank=rank,
                           dp_rank=meta.remote_dp_rank, tp_size=meta.tp_size), layers=[])
        group_by_layer = {name: i for i, names in enumerate(p['groups']) for name in names}
        transfer_names = set(worker.kv_caches)
        if (len(group_by_layer) != sum(map(len, p['groups']))
                or set(group_by_layer) != transfer_names
                or set(worker.layer_name_to_local_kv_cache_metadata) != transfer_names
                or not transfer_names.issubset(worker.layer_to_spec)):
            raise ValueError('runtime/scheduler transfer layer inventory mismatch')
        session_map = []
        for name in worker.layer_name_to_local_kv_cache_metadata:
            group = group_by_layer[name]
            indices = worker._region_session_indices(name)
            cache = worker.kv_caches[name]
            regions = []
            if worker._is_mamba_layer(name):
                if group == 0 or len(indices) != 2:
                    raise ValueError('KDA group/region mismatch')
                conv, ssm = layout.kda_conv_ssm(cache, worker.layer_to_spec[name])
                template = layout.build_mamba_offset_template(conv, ssm, worker._conv_decomp, 1)
                geometry = [(template.conv_slot_stride_bytes, list(template.conv_subprojs)),
                            (template.ssm_slot_stride_bytes, [(0, 0, template.ssm_read_bytes)])]
                kind = 'kda'
            else:
                if group != 0 or len(indices) != 1:
                    raise ValueError('attention group/region mismatch')
                geometry_info = layout.get_layer_transfer_geometry(name, cache, worker.layer_to_spec, remote.num_blocks)
                if geometry_info.transfers_per_block != 1 or geometry_info.regions_per_block != 1:
                    raise ValueError('split attention geometry not supported')
                geometry = [(geometry_info.block_stride * cache.element_size(),
                             [(0, 0, geometry_info.block_len)])]
                kind = 'attention'
            for index, (stride, segments) in zip(indices, geometry, strict=True):
                region = f'{name}/{index}'
                session = sessions[index]
                session_map.append((name, session, region))
                local_base = remote_base = 0
                if worker.kv_layer_mr_offset:
                    local_regions = worker.layer_name_to_local_kv_cache_metadata[name]
                    remote_regions = worker.layer_name_to_remote_kv_cache_metadata[engine][name]
                    region_index = indices.index(index)
                    local_mr = worker.moriio_wrapper.get_unpack_memory_metadata(local_regions[region_index])
                    remote_mr = worker.moriio_wrapper.get_unpack_memory_metadata(remote_regions[region_index])
                    # Independent registered-region coordinate system: KDA has
                    # separate conv/SSM registrations even with shared backing.
                    local_tensor = (conv, ssm)[region_index] if kind == 'kda' else cache
                    local_base = local_tensor.data_ptr() - local_mr.data
                    remote_index = worker.layer_base_addr_index[name] + region_index
                    remote_base = remote.kv_caches_base_addr[remote_index] - remote_mr.data
                    if local_base < 0 or remote_base < 0:
                        raise ValueError('registered region base exceeds layer address')
                    if kind == 'kda' and (local_base or remote_base):
                        raise ValueError('KDA requires its own conv/SSM registered base')
                    if kind == 'attention' and local_base != worker.kv_layer_mr_offset[name]:
                        raise ValueError('shared attention MR offset mismatch')
                regions.append(dict(name=region, local_stride=stride,
                                    remote_stride=stride, local_base=local_base, remote_base=remote_base,
                                    segments=segments))
            p['layers'].append(dict(name=name, kind=kind, group=group,
                                    page_tokens=p['page_tokens'], regions=regions))
        operations = sum(len(layer['regions']) for layer in p['layers'])
        # Bound no-retry events from actual layer inventory; retries are unbounded
        # in principle and remain protected by the strict event/drop limit.
        projected = self.sample_limit * (2 + 2 * operations)
        self.capacity = dict(sample_limit=self.sample_limit, operations_per_transfer=operations,
                             no_retry_event_bound=projected, max_events=self.observer.max_events,
                             retry_headroom=self.observer.max_events - projected)
        if projected >= self.observer.max_events:
            raise ValueError('runtime layer inventory exceeds event capacity')
        self.observer.attach(key, p, session_map)

    def stop_sampling(self):
        """Scheduler RPC: stop new snapshots; caller must stop new warmup admission."""
        self.sampling = False
        return dict(status='STOPPED', pending=len(self.pending))

    def close_worker(self, path=None):
        """Call ONLY on serialized model RPC after scheduler STOPPED and idle."""
        self.observer.annotations = dict(selection=copy.deepcopy(self.selection), capacity=self.capacity,
                                         selection_counts=dict(Counter(row['reason'] for row in self.selection)))
        evidence = self.close_if_idle(path)
        if evidence is None:
            return dict(status='NOT_READY', rank=self.worker.tp_rank,
                        reason='pending, incomplete or invalid observation',
                        invalid=list(self.observer.invalid), dropped=self.observer.dropped)
        evidence.update(selection=copy.deepcopy(self.selection), capacity=self.capacity)
        return dict(status='READY' if not evidence['invalid'] and not evidence['dropped'] else 'NOT_READY',
                    rank=self.worker.tp_rank, restored=not self.restores and not self.observer.restores,
                    evidence=evidence)

    def close_scheduler(self):
        if self.sampling or self.pending or self.scheduler._reqs_need_recv or self.observer.invalid:
            return dict(status='NOT_READY', pending=len(self.pending), invalid=list(self.observer.invalid))
        evidence = self.close_if_idle()
        return dict(status='READY' if evidence is not None else 'NOT_READY',
                    restored=not self.restores, metadata=copy.deepcopy(self.exported),
                    selection_counts=dict(Counter(row['reason'] for row in self.exported)))

    def close_if_idle(self, path=None):
        """Never poll or drain. Caller must hold external warmup quiescence."""
        if self.worker is not None and not any(row['sampled'] for row in self.selection):
            return None
        if self.pending or self.deferred or self.observer.dropped or self.observer.invalid:
            return None
        if self.worker is not None and self.worker._recving_transfers:
            return None
        accepted = {e['id'] for e in self.observer.events
                    if e['type'] == 'submit' and e['state'] == 'Accepted'}
        completed = {e['operation'] for e in self.observer.events
                     if e['type'] == 'completion' and e['state'] == 'Succeeded'}
        if accepted - completed:
            return None
        for obj, name, had, saved in reversed(self.restores):
            if had:
                setattr(obj, name, saved)
            else:
                delattr(obj, name)
        self.restores.clear()
        return self.observer.close(path)


def runtime_install(connector):
    """Arm ONE MoRIIOConnector in its owning process; callable after construction.

    Returns None on P (NOT_INSTALLED), otherwise Adapter retained on the connector.
    This call
    must run before warmup; worker binding wrapper is restored by close_worker.
    No environment flags or global import-time patches are installed.
    """
    import importlib.util
    from pathlib import Path
    if hasattr(connector, '_prefix_read_adapter'):
        raise ValueError('observer already installed; one mechanism window only')
    spec = importlib.util.spec_from_file_location('_prefix_read_observer_runtime',
                                                 Path(__file__).with_name('observer.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    scheduler = connector.connector_scheduler
    worker = connector.connector_worker
    if (scheduler is None) == (worker is None):
        raise ValueError('expected exactly one scheduler/worker owner')
    owner = scheduler if scheduler is not None else worker
    if owner.is_producer:
        return None  # Explicitly NOT_INSTALLED on P: no READ observer or wrappers.
    bridge = Adapter(module.Observer())
    if scheduler is not None:
        bridge.install_scheduler(scheduler)
    else:
        bridge.install_worker(worker)
        bind = connector.bind_connector_metadata
        def observed_bind(metadata):
            if not bridge.observer.enabled:
                return bind(metadata)
            result = bind(metadata)
            bridge.bind(metadata)
            return result
        bridge._replace(connector, 'bind_connector_metadata', observed_bind)
    connector._prefix_read_adapter = bridge
    return bridge
