"""Isolated complete-boundary persistence. No canonical source is edited.

Supported: sequential StandaloneRunner, exactly 3 full participants, stateless
per-fit SGD, no scheduler, no early stopping, evaluation at every boundary.
"""
import copy, hashlib, json, os, shutil, tempfile, uuid
from pathlib import Path
import torch
from federatedscope.glue.cenmag import aggregator as original
CanonicalAggregator=original.CenMagAggregator
IMPLEMENTATION_ID = 'cenmag-boundary-v1'
SCHEMA=7

def safe_state(value):
    # Evaluation metrics contain NumPy scalar wrappers. Encode primitives
    # explicitly rather than enabling arbitrary pickle globals on load.
    import numpy as np
    if isinstance(value,torch.Tensor):return value.detach().cpu().clone()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,np.ndarray):return safe_state(value.tolist())
    if isinstance(value,dict):return {safe_state(k):safe_state(v) for k,v in value.items()}
    if isinstance(value,list):return [safe_state(v) for v in value]
    if isinstance(value,tuple):return tuple(safe_state(v) for v in value)
    if type(value) in (str,int,float,bool,type(None)):return value
    raise RuntimeError('Unsupported checkpoint type: '+str(type(value)))

def digest(value):
    h=hashlib.sha256()
    def add(v):
        if isinstance(v,torch.Tensor):
            t=v.detach().cpu().contiguous();h.update(str(t.dtype).encode());h.update(str(list(t.shape)).encode());h.update(t.view(torch.uint8).numpy().tobytes())
        elif isinstance(v,dict):
            for k in sorted(v,key=lambda k:(type(k).__name__,str(k))):h.update((type(k).__name__+str(k)).encode());add(v[k])
        elif isinstance(v,(list,tuple)):
            h.update(type(v).__name__.encode())
            for x in v:add(x)
        else:h.update((type(v).__name__+repr(v)).encode())
    add(value);return h.hexdigest()

class BoundaryAggregator(CanonicalAggregator):
    def _validate_checkpoint_payload(self,payload):
        p=dict(payload);contract=p.pop('boundary_contract',None)
        # Original validator requires source version 6. Validate that inner
        # contract separately; the external schema is strictly 7.
        if int(p.get('version',-1))!=SCHEMA or contract is None:raise RuntimeError('Expected boundary schema 7')
        p['version']=original.CHECKPOINT_VERSION
        CanonicalAggregator._validate_checkpoint_payload(self,p)
        if contract.get('implementation_id')!=IMPLEMENTATION_ID:raise RuntimeError('Implementation mismatch')
        if contract.get('schema')!=SCHEMA:raise RuntimeError('Schema mismatch')
        from platform_tools.identity import compatible
        compatible(contract.get('identity', {}), self.runtime_identity, replay=getattr(self, 'portable_replay', False))
        if contract.get('contract_digest') != digest({k:v for k,v in contract.items() if k!='contract_digest'}):raise RuntimeError('Boundary runtime/identity digest mismatch')
        s=p['cenmag_state'];ids=list(self.cenmag_state.client_ids);n=int(p['server_round'])
        if s['client_ids']!=ids and tuple(s['client_ids'])!=tuple(ids):raise RuntimeError('Duplicate or missing client IDs')
        # Reject int/string duplicate IDs before canonical normalization.
        for container in [p, s]:
            for name in ['client_local_rounds','client_local_states','client_local_magnitudes']:
                keys=list(container[name]);normalized=[int(k) for k in keys]
                if len(normalized)!=len(set(normalized)):raise RuntimeError('Duplicate normalized client IDs')
        schedule=s['last_rng_schedule']
        if n>0 and (schedule.get('server_round')!=n-1 or schedule.get('completed_clients')!=ids or schedule.get('client_order')!=ids):raise RuntimeError('Partial/stale worker completion')
        if contract['runtime']['next_round']!=n or contract['runtime']['eval_completed_clients']!=ids:raise RuntimeError('Runtime/evaluation boundary mismatch')
        runtime=contract['runtime']
        required_runtime={'next_round','eval_completed_clients','server_timestamp','history_results','best_results',
            'sampler_client_state','processed_examples','client_states','optimizer_policy','scheduler',
            'dataloader_policy','early_stop_policy','queue_empty','server_cache_empty'}
        if not required_runtime.issubset(runtime):raise RuntimeError('Incomplete execution boundary metadata')
        if runtime['queue_empty'] is not True or runtime['server_cache_empty'] is not True:raise RuntimeError('Unresolved dispatcher boundary')
        if runtime['scheduler'] is not None:raise RuntimeError('Unexpected runtime scheduler')
        for name in ['processed_examples','client_states']:
            if set(runtime[name])!=set(ids):raise RuntimeError('Runtime client keys mismatch')
        steps=p['metadata']['experiment']['local_update_steps'];batch=p['metadata']['experiment']['batch_size']
        if any(v != n*steps*batch for v in runtime['processed_examples'].values()):raise RuntimeError('Processed-example counters inconsistent')
        if any(v != n-1 for v in runtime['client_states'].values()):raise RuntimeError('Client execution rounds inconsistent')
        if len(runtime['sampler_client_state'])!=len(ids)+1:raise RuntimeError('Sampler state missing')
        if n>0 and schedule['base_seed']!=p['metadata']['experiment']['seed']:raise RuntimeError('RNG base seed mismatch')

        if contract['optimizer']!={'type':'SGD','momentum':0.0,'weight_decay':0.0,'state':{},'recreated_per_fit':True}:raise RuntimeError('Unsupported optimizer contract')
        if contract['scheduler'] is not None:raise RuntimeError('Scheduler unsupported in this schema')
        for cid in ids:
            if p['client_local_rounds'][cid]!=n:raise RuntimeError('Stale client snapshot')
        weights=s['last_aggregation_weights']
        if n>0:
            if set(weights)!=set(ids) or abs(sum(weights.values())-1)>1e-10:raise RuntimeError('Invalid participation weights')
            for k in self.cenmag_state.keys.magnitude:
                mean=sum(p['client_local_magnitudes'][i][k].double()*weights[i] for i in ids)
                if float(mean.abs().max())>self.cenmag_state.tolerance:raise RuntimeError('Residuals not centered')
        signed={k:v for k,v in payload.items() if k!='boundary_contract'}
        if contract['state_digest']!=digest(signed):raise RuntimeError('Boundary state digest mismatch')

    def capture_complete_boundary(self,boundary,*,runtime,barrier_complete):
        if not barrier_complete:raise RuntimeError('Unresolved worker barrier')
        s=self.cenmag_state;ids=set(s.client_ids)
        if set(s.client_local_rounds)!=ids or s.resume_round!=boundary or any(v!=boundary for v in s.client_local_rounds.values()):raise RuntimeError('Incomplete boundary')
        if self._frozen is not None and boundary<=self._frozen['server_round']:raise RuntimeError('Stale/duplicate boundary capture')
        p=safe_state(CanonicalAggregator._build_checkpoint(self,boundary));p['version']=SCHEMA
        contract=dict(schema=SCHEMA,implementation_id=IMPLEMENTATION_ID,boundary_id=str(uuid.uuid4()),identity=copy.deepcopy(self.runtime_identity),runtime=safe_state(runtime),optimizer=dict(type='SGD',momentum=0.0,weight_decay=0.0,state={},recreated_per_fit=True),scheduler=None,state_digest=digest(p))
        contract['contract_digest']=digest(contract)
        p['boundary_contract']=contract
        self._validate_checkpoint_payload(p)
        self._frozen=copy.deepcopy(p)

    def __init__(self,*a,**kw):
        super().__init__(*a,**kw)
        self._frozen=None;self.loaded_runtime=None;self._defer_save=False;self._save_requests=[]
        cfg=self.cfg
        if str(cfg.train.optimizer.type)!='SGD' or cfg.train.optimizer.momentum!=0 or cfg.train.optimizer.weight_decay!=0:raise RuntimeError('Only canonical state-free SGD is supported')
        if int(cfg.eval.freq)!=1 or int(cfg.early_stop.patience)!=0 or cfg.train.scheduler.type!='':raise RuntimeError('Boundary v1 requires eval every round, no early stopping or scheduler')
        self.runtime_identity=copy.deepcopy(RUNTIME_IDENTITY)
        self.portable_replay=PORTABLE_REPLAY
        if not self.runtime_identity:raise RuntimeError('Execution identity must be registered before construction')

    def _build_checkpoint(self,cur_round):
        if self._frozen is None:raise RuntimeError('No coherent captured boundary')
        if int(cur_round)!=int(self._frozen['server_round']):raise RuntimeError('Requested stale/future boundary')
        return copy.deepcopy(self._frozen)

    def _verify_manifest_hash(self,path):
        path=Path(path);manifest=path.parent/'manifest.json'
        if not manifest.is_file():raise RuntimeError('Missing committed generation manifest')
        m=json.loads(manifest.read_text())
        if m.get('implementation_id')!=IMPLEMENTATION_ID or m.get('file')!=path.name or m.get('bytes')!=path.stat().st_size or m.get('sha256')!=original.checkpoint_file_sha256(path):raise RuntimeError('Checkpoint manifest/hash mismatch')

    def _read_checkpoint(self,path,verify_manifest):
        path=Path(path)
        if verify_manifest:self._verify_manifest_hash(path)
        p=torch.load(path,map_location='cpu',weights_only=True)
        self._validate_checkpoint_payload(p);return p

    def save_model(self,path,cur_round=-1):
        if self._defer_save:
            self._save_requests.append((str(path),int(cur_round)))
            return dict(status='DEFERRED_UNTIL_COMPLETE_EVALUATION_BOUNDARY')
        p=self._build_checkpoint(cur_round);self._validate_checkpoint_payload(p)
        root=Path(path).parent
        owned_root=os.environ.get('CENMAG_RUN_CHECKPOINT_ROOT')
        if owned_root and root.resolve()!=Path(owned_root).resolve():raise RuntimeError('Checkpoint output outside this new run')
        root.mkdir(parents=True,exist_ok=True)
        name=f"boundary_{p['server_round']:06d}_{p['boundary_contract']['boundary_id']}"
        destination=root/name
        if destination.exists():
            self._read_checkpoint(destination/'state.pt',True);self.last_saved_path=str(destination/'state.pt');return dict(status='PASS',path=self.last_saved_path,reused=True)
        tmp=Path(tempfile.mkdtemp(prefix='.pending_',dir=root))
        try:
            file=tmp/'state.pt'
            with file.open('wb') as f:torch.save(p,f);f.flush();os.fsync(f.fileno())
            self._read_checkpoint(file,False)
            manifest=dict(schema=SCHEMA,implementation_id=IMPLEMENTATION_ID,file='state.pt',bytes=file.stat().st_size,sha256=original.checkpoint_file_sha256(file),server_round=p['server_round'],boundary_id=p['boundary_contract']['boundary_id'])
            self._atomic_write_json(tmp/'manifest.json',manifest)
            self._verify_manifest_hash(file)
            self._sync_dir(tmp)
            os.replace(tmp,destination);self._sync_dir(root)
            pointer=root/'latest.json';old=json.loads(pointer.read_text()) if pointer.exists() else None
            self._atomic_write_json(pointer,dict(current=name,previous=old.get('current') if old else None,manifest=manifest));self._sync_dir(root)
            self.last_saved_path=str(destination/'state.pt')
            self.prune_generations(root)
            return dict(status='PASS',path=self.last_saved_path,**manifest)
        finally:
            if tmp.exists():shutil.rmtree(tmp)

    @staticmethod
    def _sync_dir(path):
        fd=os.open(path,os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(fd)
        finally:os.close(fd)

    def prune_generations(self,root):
        # Application promotes event states before retention is allowed to run.
        # No pruning from save_model: best/final duplicate calls remain harmless.
        return

    def load_model(self,path):
        if not Path(path).parent.name.startswith('boundary_'):raise RuntimeError('Uncommitted checkpoint directory rejected')
        candidate=copy.deepcopy(self._read_checkpoint(path,True))
        # Prevalidation is read-only. Roll back weights/coordinator/RNG/cache on
        # an unexpected apply failure; no partially loaded object is published.
        prior_model=copy.deepcopy(self.model.state_dict());prior_state=copy.deepcopy(self.cenmag_state.export_state());prior_rng=original.capture_rng_state();prior_cache=self._frozen
        try:
            value=CanonicalAggregator.load_model(self,path)
        except Exception:
            self.model.load_state_dict(prior_model,strict=False);self.cenmag_state.load_exported_state(prior_state);original.restore_rng_state(prior_rng);self._frozen=prior_cache;raise
        self._frozen=candidate;self.loaded_runtime=copy.deepcopy(candidate['boundary_contract']['runtime']);self.last_saved_path=str(Path(path).resolve());return value

class ControlledStop(Exception):pass

RUNTIME_IDENTITY={}
PORTABLE_REPLAY=False
def activate(identity, replay=False):
    """Explicit isolated import overlay; affects only this interpreter."""
    global RUNTIME_IDENTITY, PORTABLE_REPLAY
    PORTABLE_REPLAY=bool(replay)
    RUNTIME_IDENTITY=copy.deepcopy(identity)
    original.CenMagAggregator=BoundaryAggregator
    import importlib.util,sys
    path=Path(__file__).with_name('worker.py');name='federatedscope.glue.cenmag.worker'
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module
