"""Experiment-only gates and CPU dispatch telemetry; no production edits."""
import os
if os.environ.get('GEN_PROBE') == '1':
    import collections
    import json
    import signal
    import torch
    from pathlib import Path
    import atom.models.qwen4_exp_mono.config as config
    mono = os.environ.get('GEN_MODE') == 'mono'
    config.ENABLED_TP = (1, 8) if mono else (1,)
    config.ENABLED_WIDTHS = (1, 2, 4, 8) if mono and os.environ['GEN_TP'] == '1' else (1,)
    # runner imports ENABLED_WIDTHS by value: set config before importing it.
    from atom.models.qwen4_exp_mono.runner import QwenMonoRunner
    if os.environ.get('GEN_TP8_FINISH_PAIR') == '1':
        config.ENABLED_WIDTHS = (1, 2)
        import atom.models.qwen4_exp_mono.runner as runner_module
        runner_module.ENABLED_WIDTHS = (1, 2)
        from atom.models.qwen4_exp_mono.kernels import collective, finish, gdn_tp8
        from atom.models.qwen4_exp_mono.plan import LayerKind
        original_build_width = QwenMonoRunner._build_width
        original_kernels = QwenMonoRunner._kernels
        original_tp8_layer = QwenMonoRunner._tp8_layer
        original_ready_prepare = QwenMonoRunner.prepare
        def build_width(self, tokens):
            if self.tp != 8 or tokens != 2:
                return original_build_width(self, tokens)
            return runner_module.WidthBuild(
                gdn_layer={}, qsa_pre=None, qsa_post={},
                tp8_finish=finish._build_finish_tp8(
                    self._build_key(tokens, LayerKind.GDN),
                    paired_hc=True, shared_only=True),
                tp8_pull=collective.build_pull(tokens, 8),
                tp8_gdn_k1=(gdn_tp8.build_gdn_k1(self._build_key(tokens, LayerKind.GDN))
                            if os.environ.get('GEN_TP8_GDN_WIDTH') == '1' else None),
            )
        def kernels(build):
            if build.tp8_finish is not None and build.tp8_gdn_k2 is None:
                return [(build.tp8_finish, finish.TP8_ABI),
                        (build.tp8_pull, collective.PULL_ABI),
                        *([(build.tp8_gdn_k1, gdn_tp8.K1_ABI)]
                          if build.tp8_gdn_k1 is not None else [])]
            return original_kernels(build)
        def pair_prepare(self, tokens):
            if self.tp != 8 or tokens != 2:
                return original_ready_prepare(self, tokens)
            self.state.prepare(tokens, self.device)
            if not self.widths.prepare(tokens):
                return False
            if tokens not in self._checked:
                from atom.models.qwen4_exp_mono import contract
                contract.check_launcher(self.widths[tokens].tp8_finish,
                                        max_sgpr_spill=88, max_lds=128*1024)
                contract.check_launcher(self.widths[tokens].tp8_pull,
                                        max_sgpr_spill=0, max_lds=128*1024)
                if self.widths[tokens].tp8_gdn_k1 is not None:
                    contract.check_launcher(self.widths[tokens].tp8_gdn_k1,
                                            max_sgpr_spill=0, max_lds=128*1024)
                self._checked.add(tokens)
            return True
        def pair_layer(self, task, layer, weights, step, positions,
                       hidden_states, input_ids, pending, view):
            if self.tp != 8 or view.tokens != 2:
                return original_tp8_layer(self, task, layer, weights, step,
                                          positions, hidden_states, input_ids, pending, view)
            if pending is not None:
                self._tp8_pull(task.layer_id - 1, pending[0], view)
            if layer.ple is not None:
                if pending is not None:
                    hidden_states = layer.attn_hyper_connection.apply_pending(hidden_states, pending)
                    pending = None
                hidden_states = layer._apply_ple(hidden_states, input_ids)
            mixed, (streams, raw) = layer.attn_hyper_connection.mix_fused(hidden_states, pending)
            if task.is_gdn and self.widths[2].tp8_gdn_k1 is not None:
                bound = weights.gdn
                gdn_values = dict(x=mixed, projection_weight=bound.qkvzba,
                                  conv_weight=bound.conv, conv_state=step.conv_state,
                                  state_read=step.state_read, state_write=step.state_write,
                                  a_log=bound.a_log, dt_bias=bound.dt_bias,
                                  state=step.recurrent_state, norm_weight=bound.norm,
                                  output_weight=bound.out, partial_out=view.gdn.out,
                                  scratch=view.scratch)
                gdn_args = {key: value.data_ptr() for key, value in gdn_values.items()}
                gdn_args.update(peer_sym=self.state.peers.local, layer=task.layer_id)
                self.widths[2].tp8_gdn_k1(*gdn_tp8.K1_ABI.pack(gdn_args),
                                         stream=torch.cuda.current_stream())
                partial = view.gdn.out.contiguous()
            else:
                partial = (layer.linear_attn(mixed) if task.is_gdn
                           else layer.self_attn(positions, mixed))
            attn = runner_module.tensor_model_parallel_all_reduce(partial)
            hc, moe = weights.mlp_hc, weights.moe
            output = view.streams_out[task.layer_id & 1]
            values = dict(streams=streams, attn_out=attn, attn_raw=raw,
                          hc_norm=hc.norm, hc_down_inject=hc.down_inject, hc_up=hc.up,
                          router=moe.router, shared_w13=moe.shared_w13, shared_w2=moe.shared_w2,
                          routed_w13=moe.w13, routed_w13_scale=moe.w13_scale,
                          routed_w2=moe.w2, routed_w2_scale=moe.w2_scale,
                          expert_map=moe.expert_map, streams_out=output,
                          pending_out=view.pending_out, pending_raw=view.pending_raw,
                          moe_out=view.moe_out, scratch=view.scratch)
            packed = {key: value.data_ptr() for key, value in values.items()}
            packed.update(peer_sym=self.state.peers.local, layer=task.layer_id)
            self.widths[2].tp8_finish(*finish.TP8_ABI.pack(packed),
                                    stream=torch.cuda.current_stream())
            return output, (view.moe_out, view.pending_raw)
        QwenMonoRunner._build_width = build_width
        QwenMonoRunner._kernels = staticmethod(kernels)
        QwenMonoRunner.prepare = pair_prepare
        QwenMonoRunner._tp8_layer = pair_layer
        attention_path = 'serial-Mono-GDN/native-QSA' if os.environ.get('GEN_TP8_GDN_WIDTH') == '1' else 'native-attention'
        print(f'GEN_PATH S2=partial-Mono {attention_path} paired-HC/shared-G1/serial-G2', flush=True)
    hybrid = os.environ.get('GEN_HYBRID')
    if hybrid:
        original_layer = QwenMonoRunner._layer
        def hybrid_layer(self, task, layer, weights, step, positions, hidden_states, input_ids, pending, view):
            if self.tp == 1 and ((hybrid == 'native_qsa' and not task.is_gdn) or
                                  (hybrid == 'native_gdn' and task.is_gdn)):
                return layer(positions, hidden_states, input_ids, pending)
            return original_layer(self, task, layer, weights, step, positions, hidden_states, input_ids, pending, view)
        QwenMonoRunner._layer = hybrid_layer
    if os.environ.get('GEN_RESOURCE_AUDIT') == '1':
        from atom.models.qwen4_exp_mono import contract
        # Audit only: retain zero-private-scratch/VGPR-spill/LDS gates, log
        # scalar spill counts instead of claiming the old binary caps passed.
        original_contract = contract.check_launcher
        def audit_contract(launcher, **kwargs):
            kwargs['max_sgpr_spill'] = 4096
            resources = original_contract(launcher, **kwargs)
            print('GEN_RESOURCE_AUDIT ' + json.dumps(resources), flush=True)
            return resources
        contract.check_launcher = audit_contract
    from atom.model_engine.model_runner import ModelRunner
    from atom.utils.forward_context import get_forward_context
    counts = collections.Counter()
    prepared = {}
    original_prepare = QwenMonoRunner.prepare
    def prepare(self, tokens):
        ok = original_prepare(self, tokens)
        if str(tokens) not in prepared:
            print(f'GEN_MONO_PREPARE pid={os.getpid()} tp={self.tp} width={tokens} ok={ok}', flush=True)
        prepared[str(tokens)] = bool(ok)
        return ok
    QwenMonoRunner.prepare = prepare
    original_run = ModelRunner.run_model
    original_init = ModelRunner.__init__
    accuracy_dir = os.environ.get('GEN_ACCURACY_DIR')
    accuracy_steps = int(os.environ.get('GEN_ACCURACY_STEPS', '128'))
    accuracy_index = 0
    teacher_dir = os.environ.get('GEN_TEACHER_DIR')
    shadow_dir = os.environ.get('GEN_SHADOW_DIR')
    shadow_index = 0
    shadow_steps = int(os.environ.get('GEN_SHADOW_STEPS', '32'))
    shadow_repeat = os.environ.get('GEN_SHADOW_NATIVE_REPEAT', '1') == '1'
    shadow_layers = os.environ.get('GEN_SHADOW_LAYERS') == '1'
    if shadow_layers:
        from shadow import install_mono_trace
        install_mono_trace()
    def init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.profiler_dir = str(Path(os.environ['GEN_RUN'], 'profiles', self.rank_name))
        Path(self.profiler_dir).mkdir(parents=True, exist_ok=True)
    ModelRunner.__init__ = init
    def run(self, *args, **kwargs):
        global accuracy_index
        global shadow_index
        ctx = get_forward_context().context
        capture_accuracy = bool(accuracy_dir) and not ctx.is_dummy_run and not ctx.is_prefill
        ids = args[0] if args else kwargs.get('input_ids')
        if capture_accuracy and teacher_dir and accuracy_index < accuracy_steps:
            reference = torch.load(Path(teacher_dir, f'decode_{accuracy_index:04d}.pt'), weights_only=True)
            if tuple(ids.shape) != tuple(reference['input_ids'].shape):
                raise RuntimeError('Teacher-forcing input width differs from reference')
            if not torch.equal(ctx.positions.detach().cpu(), reference['positions']):
                raise RuntimeError('Teacher-forcing positions differ from reference')
            ids.copy_(reference['input_ids'].to(ids.device))
        accuracy_ids = ids.detach().cpu() if capture_accuracy and accuracy_index < accuracy_steps else None
        if not ctx.is_dummy_run and not ctx.is_prefill:
            key = (int(ctx.scheduled_bs), int(ctx.running_bs), bool(ctx.forward_mode.use_cudagraph))
            counts[str(key)] += 1
            total = sum(counts.values())
            if total == 1 or total % 127 == 0:
                snapshot()
        reference = None
        repeat_reference = None
        if shadow_dir and not ctx.is_dummy_run and not ctx.is_prefill and shadow_index < shadow_steps:
            from shadow import native_reference
            reference = native_reference(self, ids)
            if shadow_repeat:
                repeat_reference = native_reference(self, ids)
            if shadow_layers:
                import shadow
                shadow.active_trace = {}
        result = original_run(self, *args, **kwargs)
        if reference is not None:
            if self.rank == 0:
                from shadow import metrics
                logits, hidden = result
                payload = dict(step=shadow_index, width=int(ctx.running_bs),
                               input_ids=ids.detach().cpu(), positions=ctx.positions.detach().cpu(),
                               hidden=hidden.detach().cpu(), logits=logits.detach().cpu(),
                               reference_hidden=reference[0].detach().cpu(),
                               reference_logits=reference[1].detach().cpu(),
                               hidden_error=metrics(hidden, reference[0]),
                               logits_error=metrics(logits, reference[1]),
                               argmax_equal=bool(torch.equal(logits.argmax(-1), reference[1].argmax(-1))))
                payload['reference_top2_margin'] = reference[1].float().topk(2, dim=-1).values.diff(dim=-1).neg().detach().cpu()
                payload['argmax_matches_by_row'] = (logits.argmax(-1) == reference[1].argmax(-1)).detach().cpu()
                if repeat_reference is not None:
                    payload['native_repeat_hidden_error'] = metrics(repeat_reference[0], reference[0])
                    payload['native_repeat_logits_error'] = metrics(repeat_reference[1], reference[1])
                    payload['native_repeat_argmax_equal'] = bool(torch.equal(repeat_reference[1].argmax(-1), reference[1].argmax(-1)))
                if shadow_layers:
                    from shadow import trace_metrics
                    payload['layer_errors'] = trace_metrics(shadow.active_trace, reference[2])
                    if repeat_reference is not None:
                        payload['native_repeat_layer_errors'] = trace_metrics(repeat_reference[2], reference[2])
                destination = Path(shadow_dir)
                destination.mkdir(parents=True, exist_ok=True)
                torch.save(payload, destination / f'shadow_{shadow_index:04d}.pt')
                print('GEN_SHADOW ' + json.dumps({k: payload[k] for k in ('step','width','hidden_error','logits_error','argmax_equal')}), flush=True)
            shadow_index += 1
            if shadow_layers:
                shadow.active_trace = None
        if accuracy_dir and not ctx.is_dummy_run and ctx.is_prefill and self.rank == 0:
            destination = Path(accuracy_dir)
            destination.mkdir(parents=True, exist_ok=True)
            logits, hidden = result
            torch.save(dict(input_ids=ids.detach().cpu(), positions=ctx.positions.detach().cpu(),
                            hidden=hidden.detach().cpu(), logits=logits.detach().cpu(),
                            argmax=logits.float().argmax(-1).detach().cpu()), destination / 'prefill.pt')
        if capture_accuracy and self.rank == 0 and accuracy_index < accuracy_steps:
            logits, hidden = result
            packet = dict(input_ids=accuracy_ids, positions=ctx.positions.detach().cpu(),
                          scheduled_bs=int(ctx.scheduled_bs), running_bs=int(ctx.running_bs),
                          graph=bool(ctx.forward_mode.use_cudagraph),
                          hidden=hidden.detach().cpu(), logits=logits.detach().cpu(),
                          argmax=logits.float().argmax(-1).detach().cpu())
            destination = Path(accuracy_dir)
            destination.mkdir(parents=True, exist_ok=True)
            torch.save(packet, destination / f'decode_{accuracy_index:04d}.pt')
        if capture_accuracy:
            accuracy_index += 1
        return result
    ModelRunner.run_model = run
    def snapshot(*_):
        payload = {'pid': os.getpid(), 'dispatch_counts': dict(counts), 'mono_prepared': prepared}
        if torch.cuda.is_initialized():
            payload.update(allocated_bytes=torch.cuda.memory_allocated(), reserved_bytes=torch.cuda.memory_reserved(), peak_allocated_bytes=torch.cuda.max_memory_allocated())
        Path(os.environ['GEN_RUN'], f'telemetry_{os.getpid()}.json').write_text(json.dumps(payload, indent=2))
    signal.signal(signal.SIGUSR1, snapshot)
