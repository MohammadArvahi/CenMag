"""Non-optimizing observers for the real trainer and real aggregation path."""
import torch,time,math,copy,json
from pathlib import Path
from boundary import original,digest
from federatedscope.glue.cenmag.magnitude import binding_modules,load_global_magnitude,load_client_local_magnitude

def group(name):
    for g in ['lora_A','lora_B','client_local_magnitude','classifier']:
        if g in name:return g
    return 'frozen'

def norm(values):return math.sqrt(sum(float(v.detach().double().square().sum()) for v in values))

class Instrumentation:
    def __init__(self,root,runner,enabled=True):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True);self.runner=runner;self.enabled=enabled;self.steps=[];self.probes=[];self.aggregations=[];self.local_a={};self.local_effective={};self.overhead_seconds=0.
    def probe(self,client,stage,round_id):
        if not self.enabled:return
        start=time.perf_counter();model=client.trainer.ctx.model;modes={m:m.training for m in model.modules()};rng=original.capture_rng_state()
        samples=self.runner.data[client.ID]['probe'];data={k:torch.stack([s[k] for s in samples]).to(client.device) for k in ['input_ids','attention_mask','label']}
        try:
            model.eval()
            with torch.no_grad():out=model(input_ids=data['input_ids'],attention_mask=data['attention_mask'],labels=data['label'])
            logits=out.logits.detach().float().cpu();prob=logits.softmax(-1);pred=logits.argmax(-1);labels=data['label'].cpu()
            r=dict(round=int(round_id),client=int(client.ID),stage=stage,sample_ids=[int(s['sample_id']) for s in samples],loss=float(out.loss),logits=logits.tolist(),prediction=pred.tolist(),entropy=float((-(prob*prob.clamp_min(1e-30).log()).sum(-1)).mean()),accuracy=float((pred==labels).float().mean()),class_support=torch.bincount(labels,minlength=3).tolist(),class_predictions=torch.bincount(pred,minlength=3).tolist(),grad_enabled=False)
            prior=next((v for v in reversed(self.probes) if v['round']==r['round'] and v['client']==r['client']),None)
            if prior:
                r['logit_change_max_abs']=float((logits-torch.tensor(prior['logits'])).abs().max())
                r['prediction_change_count']=sum(x!=y for x,y in zip(r['prediction'],prior['prediction']))
                r['loss_change']=r['loss']-prior['loss']
            r['duration_seconds']=time.perf_counter()-start
            self.probes.append(r)
        finally:
            for module,mode in modes.items():module.training=mode
            original.restore_rng_state(rng)
        self.overhead_seconds+=time.perf_counter()-start
    def attach(self):
        from federatedscope.glue.cenmag.magnitude import effective_magnitude_state
        for cid,client in self.runner.client.items():
            trainer=client.trainer;backward=trainer._hook_on_batch_backward
            def fit_start(ctx,client=client):
                self.probe(client,'before_local',client.state)
            def fit_end(ctx,client=client):
                s=client.cenmag_state;model=ctx.model
                self.local_a[client.ID]=s.snapshot(model,s.keys.lora_a,fp32=False)
                self.local_effective[client.ID]={k:v.detach().cpu().clone() for k,v in effective_magnitude_state(model).items()}
                self.probe(client,'after_local',client.state);self.probe(client,'before_A_replacement',client.state)
            def step_start(ctx):ctx._step_started=time.perf_counter()
            def observed_backward(ctx,backward=backward,client=client):
                start=time.perf_counter();before={n:p.detach().clone() for n,p in ctx.model.named_parameters() if p.requires_grad};tcopy=time.perf_counter()-start
                # Core backward is called intact, including exact loss/step semantics.
                t=time.perf_counter();backward(ctx);torch.cuda.synchronize();core_seconds=time.perf_counter()-t
                t=time.perf_counter();rows={};total_sq=0.;update_sq=0.;p_m=sum(p.numel() for n,p in ctx.model.named_parameters() if 'client_local_magnitude' in n and p.requires_grad)
                for g in ['lora_A','lora_B','client_local_magnitude','classifier']:
                    params=[(n,p) for n,p in ctx.model.named_parameters() if n in before and group(n)==g]
                    gn=norm([p.grad for n,p in params if p.grad is not None]);un=norm([p.detach()-before[n] for n,p in params]);pn=norm([before[n] for n,p in params]);total_sq+=gn**2;update_sq+=un**2
                    rows[g]=dict(gradient_norm=gn,update_norm=un,parameter_norm_before=pn,relative_update_norm=un/pn if pn else None,gradient_finite=all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for n,p in params),parameter_finite=all(bool(torch.isfinite(p).all()) for n,p in params))
                mags=[(n,p) for n,p in ctx.model.named_parameters() if n in before and group(n)=='client_local_magnitude'];reg_grads=[before[n]*float(ctx.cfg.cenmag.lambda_m)/p_m for n,p in mags];task_grads=[p.grad-rg for (n,p),rg in zip(mags,reg_grads)]
                r=dict(round=int(client.state),client=client.ID,local_step=int(ctx.cur_batch_i),epoch=int(ctx.cur_epoch_i),loss_before_step=float(ctx.loss_task.detach()),task_loss=float(ctx.loss_batch.detach()),regularization_loss=float(ctx.loss_regular.detach()),gradient_norm=math.sqrt(total_sq),update_norm=math.sqrt(update_sq),lr=[g['lr'] for g in ctx.optimizer.param_groups],groups=rows,regularization_gradient_norm=norm(reg_grads),magnitude_task_gradient_norm=norm(task_grads),residual_norm_before=norm([before[n] for n,p in mags]),residual_norm_after=norm([p.detach() for n,p in mags]),effective_magnitude_norm=norm([b.global_magnitude+b.client_local_magnitude for b in binding_modules(ctx.model).values()]),center_norm=norm([b.global_magnitude for b in binding_modules(ctx.model).values()]),sample_ids=ctx.data_batch['sample_id'].tolist(),allocated_bytes=torch.cuda.memory_allocated(),reserved_bytes=torch.cuda.memory_reserved(),core_backward_step_seconds=core_seconds)
                r['step_duration_seconds']=time.perf_counter()-ctx._step_started
                r['relative_effective_magnitude_update']=rows['client_local_magnitude']['update_norm']/max(r['effective_magnitude_norm'],1e-30)
                r['regularization_to_task_gradient_ratio']=r['regularization_gradient_norm']/max(r['magnitude_task_gradient_norm'],1e-30)
                r['observer_seconds']=tcopy+time.perf_counter()-t;self.overhead_seconds+=r['observer_seconds'];self.steps.append(r)
                with (self.root/'steps.jsonl').open('a') as f:f.write(json.dumps(r)+'\n')
            if self.enabled:
                trainer.register_hook_in_train(step_start,'on_batch_start',insert_pos=-1)
                trainer.replace_hook_in_train(observed_backward,'on_batch_backward','_hook_on_batch_backward')
                trainer.register_hook_in_train(fit_start,'on_fit_start',insert_pos=-1)
                trainer.register_hook_in_train(fit_end,'on_fit_end',insert_pos=-1)
    def after_aggregation(self,old_center,old_a,round_id,uploads):
        if not self.enabled:return
        start=time.perf_counter();nested_before=self.overhead_seconds;server=self.runner.server;s=server.cenmag_state;model=server.model
        resident=copy.deepcopy(model.state_dict());old_buffers={k:b.global_magnitude.detach().clone() for k,b in binding_modules(model).items()}
        try:
            center_drift=norm([s.global_magnitude[k]-old_center[k] for k in old_center]);global_a_change=norm([s.global_a[k]-old_a[k] for k in old_a]);rows={}
            for cid,client in self.runner.client.items():
                local=self.local_a[cid];replacement=norm([s.global_a[k]-local[k] for k in local]);functional_sq=0.
                for ak in local:
                    bk=ak.replace('lora_A','lora_B');b=s.client_local_states[cid][bk].double();d=(s.global_a[ak]-local[ak]).double();functional_sq+=float(((b.T@b)*(d@d.T).T).sum())*4 # scaling=(16/8)=2
                s.load_client_local_state(model,cid);s.load_global_a(model)
                load_global_magnitude(model,old_center);load_client_local_magnitude(model,{k:self.local_effective[cid][k]-old_center[k] for k in old_center})
                self.probe(client,'after_A_replacement',round_id)
                s.load_effective_magnitude(model,cid);self.probe(client,'after_recenter',round_id)
                rows[cid]=dict(local_to_global_A_norm=replacement,pre_normalization_B_conditioned_delta_V_norm=math.sqrt(max(functional_sq,0)),residual_norm=norm(s.client_local_magnitudes[cid].values()),effective_magnitude_norm=norm(s.effective_magnitude(cid).values()),reconstruction_max_abs=max(float((s.effective_magnitude(cid)[k]-self.local_effective[cid][k]).abs().max()) for k in old_center))
            # Independent processed-example weighted reference, before uploads mutate.
            total=sum(v[0] for v in uploads.values());error=max(float((s.global_a[k]-sum(uploads[i][1][k].cpu()*uploads[i][0]/total for i in uploads)).abs().max()) for k in s.global_a)
            self.aggregations.append(dict(round=round_id,center_drift_norm=center_drift,global_A_update_norm=global_a_change,aggregation_max_abs_A_error=error,weights=s.last_aggregation_weights,centering=s.last_centering,validation=s.last_aggregation_validation,clients=rows))
            assert error<1e-6
            assert all(r['reconstruction_max_abs']<1e-6 for r in rows.values())
        finally:
            model.load_state_dict(resident,strict=False);load_global_magnitude(model,old_buffers)
        self.overhead_seconds+=(time.perf_counter()-start)-(self.overhead_seconds-nested_before)
    def save(self):
        (self.root/'instrumentation.json').write_text(json.dumps(dict(steps=self.steps,probes=self.probes,aggregations=self.aggregations,observer_seconds=self.overhead_seconds),indent=2))
