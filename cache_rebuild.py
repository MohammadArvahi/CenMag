"""Recreate canonical MNLI tensors from verified upstream bytes; CPU only."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import sys
import time
import uuid


def file_sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def tensor_identity(tensor):
    a=tensor.detach().cpu().contiguous().numpy();view=memoryview(a).cast('B');h=hashlib.sha256()
    for off in range(0,len(view),8*1024**2):h.update(view[off:off+8*1024**2])
    return dict(dtype=str(tensor.dtype),shape=list(tensor.shape),sha256=h.hexdigest())


def rebuild(release,root,plan_path):
    release,root=Path(release).resolve(),Path(root).resolve()
    target=root/'cache'
    if target.exists():raise RuntimeError('Existing cache is never replaced')
    plan=json.loads(Path(plan_path).read_text())
    for record in plan['download_records']:
        if record['group'] not in ['model','dataset','metric']:continue
        p=root/record['relative_path']
        if not p.is_file() or p.stat().st_size!=record['bytes'] or file_sha(p)!=record['sha256']:
            raise RuntimeError('Source asset identity mismatch: '+record['relative_path'])
    os.environ.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
    sys.path[:0]=[str(release/'vendor'),str(release/'runtime')]
    import torch
    import numpy as np
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    from federatedscope.core.splitters.utils import dirichlet_distribution_noniid_slice
    started=time.monotonic();torch.set_num_threads(2)
    tables={s:pq.read_table(root/'dataset/mnli'/f'{n}-00000-of-00001.parquet').to_pydict() for s,n in [('train','train'),('val','validation_matched')]}
    y=np.asarray(tables['train']['label']);vy=np.asarray(tables['val']['label'])
    # The same canonical call order, seed, prior and splitter are used verbatim.
    saved_rng=np.random.get_state()
    try:
        np.random.seed(12345)
        part=dirichlet_distribution_noniid_slice(y,3,.5)
        vp=dirichlet_distribution_noniid_slice(vy,3,.5,prior=[y[p].tolist() for p in part])
    finally:np.random.set_state(saved_rng)
    manifest={'partition_seed':12345,'alpha':.5,'labels':{'0':'entailment','1':'neutral','2':'contradiction'},'source_splits':['train','validation_matched'],'test_used':False,'clients':{}}
    def array_hash(a):return hashlib.sha256(np.asarray(a,dtype='<i8').tobytes()).hexdigest()
    for i in range(3):
        manifest['clients'][str(i+1)]={s:{'count':len(ix),'indices_sha256':array_hash(ix),'class_counts':np.bincount(labels[ix],minlength=3).tolist()} for s,ix,labels in [('train',part[i],y),('val',vp[i],vy)]}
    if manifest!=plan['reference_partition']:raise RuntimeError('Canonical six-array partition manifest mismatch; do not substitute a new partition')
    tokenizer=AutoTokenizer.from_pretrained(root/'model',local_files_only=True)
    payload={}
    for split,table in tables.items():
        blocks=[]
        for off in range(0,len(table['label']),4096):
            encoded=tokenizer(table['premise'][off:off+4096],table['hypothesis'][off:off+4096],padding='max_length',max_length=128,truncation=True,return_tensors='pt')
            blocks.append(encoded)
        payload[split]={k:torch.cat([b[k] for b in blocks]) for k in blocks[0]}
        payload[split]['label']=torch.tensor(table['label'])
        payload[split]['sample_id']=torch.arange(len(table['label']))
        del blocks
    payload['partition']={i+1:{'train':torch.tensor(part[i]),'val':torch.tensor(vp[i])} for i in range(3)}
    tensors={s:{k:tensor_identity(t) for k,t in payload[s].items()} for s in ['train','val']}
    partitions={str(cid):{s:tensor_identity(t) for s,t in p.items()} for cid,p in payload['partition'].items()}
    if tensors!=plan['canonical_tensors'] or partitions!=plan['canonical_partitions']:
        raise RuntimeError('Full canonical token/label/sample-ID/partition tensor identity mismatch')
    temp=root/('.cache_pending_'+uuid.uuid4().hex);temp.mkdir()
    # Preserve original basename/key order/serialization settings for exact file identity.
    torch.save(payload,temp/'data.pt')
    (temp/'partition.json').write_text(json.dumps(manifest,indent=2)+'\n')
    files={n:dict(bytes=(temp/n).stat().st_size,sha256=file_sha(temp/n)) for n in ['data.pt','partition.json']}
    exact=all(files[n]['sha256']==v['sha256'] for n,v in plan['reference_cache'].items())
    report=dict(status='PASS' if exact else 'FAIL',scope='CPU_CANONICAL_MNLI_RECONSTRUCTION_NO_TRAINING',seconds=time.monotonic()-started,
                all_six_partition_arrays_equal=True,all_token_label_sample_id_tensors_equal=True,
                exact_original_cache_file_hashes=exact,files=files,tensors=tensors,partitions=partitions,
                tokenizer_classes=dict(fast=tokenizer.is_fast,class_name=type(tokenizer).__name__),
                source_parquet_sha256={r['relative_path']:r['sha256'] for r in plan['download_records'] if r['group']=='dataset'})
    (temp/'rebuild_validation.json').write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
    if not exact:
        raise RuntimeError('Tensor identities match but original cache serialization hash differs; evidence retained at '+str(temp)+'; runner compatibility remains blocked')
    temp.rename(target)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['release','root','plan']:p.add_argument('--'+name,required=True)
    args=p.parse_args()
    try:print(json.dumps(rebuild(args.release,args.root,args.plan),sort_keys=True,indent=2))
    except Exception as e:print(json.dumps(dict(status='FAIL',error=str(e))),file=sys.stderr);sys.exit(1)
