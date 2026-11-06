"""One fresh process, one GPU, three sequential canonical clients."""
from pathlib import Path
import copy
import json
import logging
import os
import signal
import sys
import time
import traceback

RELEASE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(RELEASE), str(RELEASE / 'vendor'), str(RELEASE / 'runtime')]


def make_cfg(rounds, out, assets, resume=''):
    from federatedscope.core.configs.config import global_cfg
    cfg = global_cfg.clone()
    cfg.merge_from_other_cfg(cfg.load_cfg((RELEASE / 'configs/mnli.json').read_text()))
    cfg.offline.model_dir = str(Path(assets['model']).resolve())
    cfg.offline.dataset_dir = str(Path(assets['dataset']).resolve())
    cfg.offline.metric_dir = str(Path(assets['metric']).resolve())
    cfg.data.num_labels = 3
    cfg.data.label_list = ['entailment', 'neutral', 'contradiction']
    cfg.data.root = str(out / 'cache/datasets')
    cfg.llm.cache.model = str(out / 'cache/model')
    cfg.outdir = str(out)
    cfg.federate.save_to = str(out / 'checkpoints/requested.pt')
    cfg.federate.restore_from = str(resume)
    cfg.federate.total_round_num = rounds
    cfg.federate.save_freq = 1
    cfg.eval.report = ['weighted_avg', 'avg', 'raw']
    cfg.cenmag.validation_mode = False
    return cfg


def run(args, *, cfg, asset_identity, checkpoint_validator, session_class):
    from platform_tools.common import atomic, read
    from platform_tools.environment import inspect
    from platform_tools.identity import make, compatible
    out = Path(args.out)
    assets = read(args.assets)
    operations = read(RELEASE / 'configs/operations.json')
    environment = inspect(out, cuda=True)
    atomic(out / 'environment.json', environment)
    if environment['core_mismatches']:
        raise RuntimeError('Dependency profile mismatch; inspect environment.json; no substitution is automatic')
    if environment['visible_devices'] != 1 or not environment['cuda_available']:
        raise RuntimeError('Exactly one CUDA device is required; select accelerator before execution')
    import torch
    import numpy as np
    import yaml
    from boundary import activate, ControlledStop, safe_state, digest
    from runner import BoundaryRunner
    from data_runtime import load_data
    from observability import Observer
    from federatedscope.core.auxiliaries.utils import setup_seed
    config_copy = copy.deepcopy(cfg)
    config_copy.clear_aux_info()
    (out / 'resolved.yaml').write_text(yaml.safe_dump(safe_state(config_copy)))
    identity = make(config_copy, asset_identity, environment)
    atomic(out / 'identity.json', identity)
    if args.checkpoint:
        checkpoint_validator(Path(args.checkpoint), identity)
    worker = activate(identity, replay=args.replay)
    setup_seed(cfg.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    data, selection = load_data(assets['cache'], cfg, None)
    atomic(out / 'selection.json', selection)
    def stop(signum, frame):
        os.environ['CENMAG_STOP_REQUESTED'] = '1'
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    observer = None
    session = None
    started = time.perf_counter()
    result = dict(status='RUNNING', next_round=None, external_verified=False, requested_stop_at=args.stop_at)
    try:
        runner = BoundaryRunner(data=data, server_class=worker.CenMagServer, client_class=worker.CenMagClient, config=cfg)
        server = runner.server
        server._boundary_clients = runner.client
        server._processed_examples = {i: 0 for i in [1, 2, 3]}
        runtime = server.aggregator.loaded_runtime
        if runtime:
            server.history_results = copy.deepcopy(runtime['history_results'])
            server.best_results = copy.deepcopy(runtime['best_results'])
            server.sampler.client_state = np.asarray(runtime['sampler_client_state'])
            server._processed_examples = copy.deepcopy(runtime['processed_examples'])
            server.cur_timestamp = runtime['server_timestamp']
            for cid, client in runner.client.items():
                client.state = runtime['client_states'][cid]
        observer = Observer(out, runner, operations, args.diagnostics)
        observer.attach()
        session = session_class(out, observer, operations, args.stop_at, identity,
                          (runtime or {}).get('research_platform'))
        handle = runner._handle_msg
        def observed_message(message, *a, **kw):
            if message.msg_type == 'model_para' and message.receiver == [0]:
                server._processed_examples[int(message.sender)] += int(message.content[0])
            return handle(message, *a, **kw)
        runner._handle_msg = observed_message
        aggregate = server._perform_federated_aggregation
        def observed_aggregation():
            state = server.cenmag_state
            old_center, old_a = copy.deepcopy(state.global_magnitude), copy.deepcopy(state.global_a)
            n = int(server.state)
            uploads = copy.deepcopy(server.msg_buffer['train'][n])
            result = aggregate()
            observer.after_aggregation(old_center, old_a, n, uploads)
            return result
        server._perform_federated_aggregation = observed_aggregation
        if args.checkpoint:
            atomic(out / 'replay_compatibility.json', compatible(server.aggregator._frozen['boundary_contract']['identity'], identity, args.replay))
        session.before_dispatch(server)
        try:
            runner.run()
            result['termination'] = 'normal'
        except ControlledStop as error:
            result['termination'] = str(error)
        state = server.cenmag_state
        state.assert_client_local_isolation()
        state.assert_detached_from_live_model(server.model)
        completed = int(server.aggregator._frozen['server_round']) if server.aggregator._frozen else int(server.state)
        result.update(status='PASS' if completed >= args.stop_at else 'STOPPED_COHERENTLY', next_round=completed,
                      target_reached=completed >= args.stop_at,
                      checkpoint=getattr(server.aggregator, 'last_saved_path', None),
                      processed_examples=server._processed_examples)
    except BaseException as error:
        result.update(status='FAIL', traceback=traceback.format_exc(),
                      numerical_failure=isinstance(error, FloatingPointError)
                      or 'non-finite' in str(error).lower())
        raise
    finally:
        if observer:
            atomic(out / 'instrumentation_cost.json', dict(observer_seconds=observer.overhead_seconds,
                   forensic_seconds=observer.deep.overhead_seconds, total_seconds=time.perf_counter()-started,
                   overhead_comparison_status='UNMEASURED_REQUIRES_MATCHED_OFF_LIGHT_FORENSIC_RUNS'))
        result['total_seconds'] = time.perf_counter() - started
        if torch.cuda.is_available():
            result['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
            result['peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
        atomic(out / 'result.json', result)
        print(json.dumps(result, sort_keys=True), flush=True)
    return result
