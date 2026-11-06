"""Canonical tokenizer and LDA split; optional predeclared validation subset."""
from pathlib import Path
import json,hashlib
import numpy as np
import torch
from torch.utils.data import Dataset,Subset

def array_hash(v):return hashlib.sha256(np.asarray(v,dtype='<i8').tobytes()).hexdigest()
class Rows(Dataset):
 def __init__(self,tensors):self.tensors=tensors
 def __len__(self):return len(self.tensors['label'])
 def __getitem__(self,i):return {k:v[i] for k,v in self.tensors.items()}

def selected_indices(indices,labels,n):
 # Proportional stratified subset: largest remainders, original partition order within class.
 ix=indices.numpy();ys=labels[indices].numpy()
 if n is None or n>=len(ix):return indices
 counts=np.bincount(ys,minlength=3);raw=counts*n/len(ix);take=np.floor(raw).astype(int)
 for k in np.argsort(-(raw-take),kind='stable')[:n-int(take.sum())]:take[k]+=1
 chosen=set(int(v) for k in range(3) for v in ix[ys==k][:take[k]])
 return torch.tensor([int(v) for v in ix if int(v) in chosen])
def probe_indices(indices,labels,n):
 # Fixed validation diagnostics include every represented class, even rare ones.
 ix=indices.numpy();ys=labels[indices].numpy();counts=np.bincount(ys,minlength=3)
 if n>=len(ix):return indices
 present=counts>0
 if n<int(present.sum()):raise ValueError('Probe too small to represent observed classes')
 take=present.astype(int);remaining=n-int(take.sum());raw=counts*remaining/len(ix)
 take+=np.floor(raw).astype(int)
 for k in np.argsort(-(raw-np.floor(raw)),kind='stable')[:n-int(take.sum())]:take[k]+=1
 take=np.minimum(take,counts)
 while take.sum()<n:
  k=int(np.argmax(counts-take));take[k]+=1
 chosen=set(int(v) for k in range(3) for v in ix[ys==k][:take[k]])
 return torch.tensor([int(v) for v in ix if int(v) in chosen])

def load_data(cache,cfg,evaluation_counts=None):
 from federatedscope.core.auxiliaries.dataloader_builder import get_dataloader
 payload=torch.load(Path(cache)/'data.pt',weights_only=True,map_location='cpu');data={};selection={}
 for cid,part in payload['partition'].items():
  train=part['train'];val=selected_indices(part['val'],payload['val']['label'],None if evaluation_counts is None else evaluation_counts[cid-1])
  data[cid]={s:get_dataloader(Subset(Rows(payload[s]),ix.tolist()),cfg,split=s) for s,ix in [('train',train),('val',val)]}
  probe=probe_indices(part['val'],payload['val']['label'],32)
  data[cid]['probe']=[Rows(payload['val'])[int(i)] for i in probe]
  selection[cid]={'train_count':len(train),'val_count':len(val),'train_indices_sha256':array_hash(train),'val_indices_sha256':array_hash(val),'val_indices':val.tolist(),'val_class_counts':torch.bincount(payload['val']['label'][val],minlength=3).tolist(),'probe_indices':probe.tolist(),'probe_indices_sha256':array_hash(probe)}
 return data,selection
