"""Train only inserted residual blocks and head; stop at the first validated improvement."""
import argparse,copy,hashlib,itertools,json,math,random,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from training_core import seed_worker
from model import ImageModel,MODEL_NAMES
from graft import STUDENT,initialize,training_mode,frozen_state,assert_frozen
from prepare import prepare
from training_core import CachedDataset,loader,validate,atomic_save
from utils import PROJECT_DIR,MODEL_PATH,RECIPE,calculate_metrics,load_model,model_memory_format
SOURCE_SHA='f7d35a6bb77819750d4b59fdeac0f273b775578ed8ec1bf39e6bafa525a7f6f5'
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
class IndexedDataset(CachedDataset):
    def __getitem__(self,i):
        x,y=super().__getitem__(i)
        return x,y,i

class StratifiedBatches:
    def __init__(self,labels,batch_size,seed):
        rng=np.random.default_rng(seed);n=len(labels);count=math.ceil(n/batch_size)
        positive=rng.permutation(np.flatnonzero(labels==1)).tolist();negative=rng.permutation(np.flatnonzero(labels==0)).tolist()
        assert len(positive)>=count,'Need at least one positive per batch'
        capacities=[batch_size]*(count-1)+[n-batch_size*(count-1)]
        batches=[[positive[i]] for i in range(count)]
        for j,index in enumerate(positive[count:]):
            slot=j%count
            while len(batches[slot])>=capacities[slot]:slot=(slot+1)%count
            batches[slot].append(index)
        cursor=0
        for i,batch in enumerate(batches):
            needed=capacities[i]-len(batch);batch.extend(negative[cursor:cursor+needed]);cursor+=needed;rng.shuffle(batch)
        flat=[i for batch in batches for i in batch]
        assert len(flat)==n and len(set(flat))==n and cursor==len(negative)
        self.batches=batches
    def __iter__(self):return iter(self.batches)
    def __len__(self):return len(self.batches)

def select(probabilities,labels,allowed):
    metrics={}
    options=[]
    names=[n for n in MODEL_NAMES if n in allowed]
    for count in range(1,len(names)+1):
        for subset in itertools.combinations(names,count):
            key='+'.join(subset)
            metrics[key]=calculate_metrics(labels,np.mean([probabilities[n] for n in subset],axis=0))
            options.append((key,list(subset)))
    key,chosen=max(options,key=lambda pair:metrics[pair[0]]['auroc'])
    return {'model_names':chosen,'selected_candidate':key,'validation':metrics[key],
        'candidate_metrics':metrics,'metric':'official_validation_auroc','threshold':.5,
        'tta':False,'inference_precision':'FP32','rule':'predeclared equal-mean subsets; student eligible only after a full training epoch'}

@torch.inference_mode()
def teacher_targets(source,allowed,indices,run_dir):
    path=run_dir/'teacher_targets.npy';meta=path.with_suffix('.json')
    if path.exists() and meta.exists():
        record=json.loads(meta.read_text())
        assert record['source_sha256']==SOURCE_SHA and record['sha256']==digest(path)
        return torch.from_numpy(np.load(path)).to('cuda')
    old=PROJECT_DIR/'runs/20261025_frozen_head'
    original=torch.load(old/'baseline_snapshot/model/best_model.pth',map_location='cpu',weights_only=True)
    probs=[];provenance={}
    for name in allowed:
        state=source['models'][name]['state_dict']
        assert source['models'][name]['model_config']['image_size']==256
        assert all(torch.equal(v,original['models'][name]['state_dict'][k]) for k,v in state.items()
            if not k.startswith(('network.fc.','network.classifier.')))
        feature=old/'features'/f'{name}_train.npy';info=json.loads(feature.with_suffix('.json').read_text())
        assert info['sha256']==digest(feature) and len(indices)==info['shape'][0]
        data=np.load(feature,mmap_mode='r');prefix='network.fc.1.' if name=='resnet18' else 'network.classifier.1.'
        w=state[prefix+'weight'].to('cuda');b=state[prefix+'bias'].to('cuda');values=[]
        for start in range(0,len(data),512):
            x=torch.from_numpy(np.array(data[start:start+512])).to('cuda')
            values.append(torch.softmax(F.linear(x,w,b),1).cpu().numpy())
        probs.append(np.concatenate(values));provenance[name]=info['sha256']
        del w,b
    probability=np.clip(np.mean(probs,axis=0),1e-6,1-1e-6)
    softened=torch.softmax(torch.from_numpy(np.log(probability))/2.,1).numpy()
    assert len(softened)==len(indices) and np.isfinite(softened).all()
    np.save(path,softened)
    meta.write_text(json.dumps({'source_sha256':SOURCE_SHA,'sha256':digest(path),'temperature':2.,
        'training_only':True,'test_or_validation_targets_used':False,'feature_sha256':provenance},indent=2),encoding='utf-8')
    return torch.from_numpy(softened).to('cuda')

def state_entry(student,metrics,epoch):
    return {'state_dict':{k:v.detach().cpu().clone() for k,v in student.state_dict().items()},
        'model_config':dict(student.model_config),'validation':metrics,'epoch':epoch,
        'training_role':'inserted residual blocks and classifier; inherited blocks frozen',
        'student_trained_epochs':epoch}

def run(args):
    torch.set_num_threads(4);torch.manual_seed(args.seed);random.seed(args.seed);np.random.seed(args.seed)
    if not torch.cuda.is_available():raise RuntimeError('GPU required')
    torch.cuda.manual_seed_all(args.seed);torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    run_dir=PROJECT_DIR/'runs'/args.run_name
    source_path=run_dir/'baseline_snapshot/model/best_model.pth';assert digest(source_path)==SOURCE_SHA
    source=torch.load(source_path,map_location='cpu',weights_only=True)
    baseline_auc=source['inference_selection']['validation']['auroc']
    allowed=source['inference_selection']['model_names']
    config={**vars(args),'recipe':RECIPE,'source_checkpoint_sha256':SOURCE_SHA,'baseline_auroc':baseline_auc,
        'train_scope':'8 inserted residual blocks and classifier; all inherited blocks and teacher branches frozen',
        'external_pretrained_weights_used':False,'distillation_temperature':2.,'hard_ce_weight':.35,'kd_weight':.15,'pairwise_weight':.5,'stratified_without_replacement':True,
        'input_size':256,'training_precision':'AMP FP16','inference_precision':'FP32',
        'early_success_rule':'stop after first complete epoch with independently validated AUROC > baseline + min_delta',
        'source_files_sha256':{p.name:digest(p) for p in Path(__file__).parent.glob('*.py')}}
    cp=run_dir/'config.json'
    if cp.exists():
        if not args.resume:raise FileExistsError('Use --resume')
        previous=json.loads(cp.read_text())
        for k in ['recipe','source_checkpoint_sha256','max_epochs','batch_size','seed','learning_rate','head_lr','min_delta']:
            assert config[k]==previous[k],k
    cp.write_text(json.dumps(config,indent=2),encoding='utf-8')
    audit=prepare();train=IndexedDataset('train',True,audit['train_indices_excluded_exact_validation_duplicates'])
    valid=CachedDataset('valid');vb=loader(valid,64,args.workers)
    teacher=teacher_targets(source,allowed,train.indices,run_dir)
    student,prefixes,mapping=initialize(source['models']['resnet18']['state_dict'])
    warm=torch.load(run_dir/'baseline_snapshot/student_initialization.pth',map_location='cpu',weights_only=True)
    student.load_state_dict(warm['state_dict'])
    frozen=frozen_state(student,prefixes)
    student.to(device='cuda',memory_format=model_memory_format(STUDENT))
    new_params=[p for n,p in student.named_parameters() if n.startswith(tuple(prefixes))]
    optimizer=torch.optim.AdamW([{'params':new_params,'lr':args.learning_rate,'role':'new_blocks'},
        {'params':student.network.fc.parameters(),'lr':args.head_lr,'role':'head'}],weight_decay=1e-4)
    scaler=torch.amp.GradScaler('cuda',init_scale=1024.)
    labels=train.labels[train.indices];positive=int(labels.sum())
    criterion=nn.CrossEntropyLoss(weight=torch.tensor([1.,math.sqrt((len(labels)-positive)/positive)],device='cuda'))
    bundle=copy.deepcopy(source)
    for name in bundle['models']:
        bundle['models'][name]['model_config']['recipe']=RECIPE
        bundle['models'][name]['training_role']='unchanged legacy teacher'
    baseline_prob={}
    if not (run_dir/'baseline_metrics.json').exists():
        for name in allowed:
            model=ImageModel(name);model.load_state_dict(source['models'][name]['state_dict'])
            model.to(device='cuda',memory_format=model_memory_format(name))
            metrics,prob=validate(model,vb,torch.device('cuda'));baseline_prob[name]=prob
            bundle['models'][name]['validation']=metrics
            np.save(run_dir/f'{name}_teacher_probabilities.npy',prob);del model
        metrics,prob=validate(student,vb,torch.device('cuda'))
        assert abs(metrics['auroc']-warm['validation']['auroc'])<1e-9
        bundle['models'][STUDENT]=state_entry(student,metrics,0)
        bundle['inference_selection']=select(baseline_prob,valid.labels,allowed)
        bundle['training_complete']=False;atomic_save(bundle,MODEL_PATH)
        (run_dir/'baseline_metrics.json').write_text(json.dumps(bundle['inference_selection'],indent=2),encoding='utf-8')
        (run_dir/'graft_mapping.json').write_text(json.dumps({'mapping':mapping,'trainable_prefixes':prefixes,'gpu_fp32_initial_function_match':True},indent=2),encoding='utf-8')
    else:
        baseline_prob={n:np.load(run_dir/f'{n}_teacher_probabilities.npy') for n in allowed}
    history=[];start_epoch=0;resume_batch=0;running_count=0;running_loss=0.
    last=run_dir/'student_last.pth';best_path=run_dir/'student_best.pth';success=False;budget_stop=False
    if last.exists():
        saved=torch.load(last,map_location='cpu',weights_only=False)
        student.load_state_dict(saved['student']);optimizer.load_state_dict(saved['optimizer']);scaler.load_state_dict(saved['scaler'])
        start_epoch=saved['next_epoch'];history=saved['history'];resume_batch=saved.get('resume_batch',0)
        running_count=saved.get('running_count',0);running_loss=saved.get('running_loss',0.)
        torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state_all(saved['cuda_rng'])
    def save_last(next_epoch,batch=0,count=0,total_loss=0.):
        atomic_save({'student':student.state_dict(),'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict(),
            'next_epoch':next_epoch,'resume_batch':batch,'running_count':count,'running_loss':total_loss,'history':history,
            'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()},last)
    for epoch in range(start_epoch,args.max_epochs):
        training_mode(student,prefixes)
        factor=.1+.9*(1+math.cos(math.pi*epoch/max(1,args.max_epochs-1)))/2
        optimizer.param_groups[0]['lr']=args.learning_rate*factor;optimizer.param_groups[1]['lr']=args.head_lr*factor
        sampler=StratifiedBatches(labels,args.batch_size,args.seed+epoch)
        batches=DataLoader(train,batch_sampler=sampler,num_workers=args.workers,pin_memory=True,persistent_workers=args.workers>0,worker_init_fn=seed_worker,generator=torch.Generator().manual_seed(args.seed+epoch))
        total_loss=running_loss if epoch==start_epoch else 0.;count=running_count if epoch==start_epoch else 0
        started=time.perf_counter()
        for step,(x,y,index) in enumerate(batches,1):
            if epoch==start_epoch and step<=resume_batch:continue
            x=x.to('cuda',non_blocking=False,memory_format=model_memory_format(STUDENT));y=y.to('cuda');index=index.to('cuda')
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.float16):
                logits=student(x);hard=criterion(logits,y)
            kd=F.kl_div(F.log_softmax(logits.float()/2.,1),teacher[index],reduction='batchmean')*4.
            scores=logits.float()[:,1]-logits.float()[:,0]
            pos=scores[y==1];neg=scores[y==0];assert len(pos)>0 and len(neg)>0
            ranking=F.softplus(neg[None,:]-pos[:,None]).mean()
            loss=.5*ranking+.35*hard+.15*kd
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite distillation loss')
            scaler.scale(loss).backward();scaler.unscale_(optimizer);nn.utils.clip_grad_norm_([p for p in student.parameters() if p.requires_grad],5.)
            scaler.step(optimizer);scaler.update();count+=len(y);total_loss+=float(loss.detach())*len(y)
            if step==1 or step%100==0 or step==len(batches):
                print(f'STUDENT epoch={epoch+1}/{args.max_epochs} batch={step}/{len(batches)} loss={total_loss/count:.5f} elapsed={time.perf_counter()-started:.1f}s',flush=True)
            if (run_dir/'STOP_AFTER_EPOCH').exists():
                assert_frozen(student,frozen);save_last(epoch,step,count,total_loss);budget_stop=True;break
        del batches
        if budget_stop:break
        assert count==len(train);assert_frozen(student,frozen)
        metrics,prob=validate(student,vb,torch.device('cuda'))
        selection=select({**baseline_prob,STUDENT:prob},valid.labels,[*allowed,STUDENT])
        champion=torch.load(MODEL_PATH,map_location='cpu',weights_only=True)
        before=champion['inference_selection']['validation']['auroc']
        if selection['validation']['auroc']>before+1e-12:
            champion['models'][STUDENT]=state_entry(student,metrics,epoch+1)
            champion['inference_selection']=selection;atomic_save(champion,MODEL_PATH)
        previous_best=torch.load(best_path,map_location='cpu',weights_only=True)['validation']['auroc'] if best_path.exists() else -1.
        if metrics['auroc']>previous_best:atomic_save(state_entry(student,metrics,epoch+1),best_path)
        row={'epoch':epoch+1,'train_samples':count,'train_loss':total_loss/count,'epoch_seconds':time.perf_counter()-started,
            **metrics,'best_global_auroc':max(before,selection['validation']['auroc']),'inherited_state_unchanged':True}
        history.append(row);pd.DataFrame(history).to_csv(run_dir/'student_history.csv',index=False)
        save_last(epoch+1);resume_batch=0;running_count=0;running_loss=0.
        print('EPOCH_RESULT '+json.dumps(row),flush=True)
        if selection['validation']['auroc']>baseline_auc+args.min_delta:
            success=True;print('FIRST_IMPROVEMENT_STOP',flush=True);break
    if not history:
        (run_dir/'BUDGET_STOPPED.json').write_text(json.dumps({'completed_epochs':0,'partial_checkpoint_saved':True}),encoding='utf-8');return
    del student,optimizer,scaler,teacher
    torch.cuda.empty_cache()
    champion=torch.load(MODEL_PATH,map_location='cpu',weights_only=True)
    if STUDENT not in champion['inference_selection']['model_names']:
        champion['models'][STUDENT]=torch.load(best_path,map_location='cpu',weights_only=True)
        atomic_save(champion,MODEL_PATH)
    actual={}
    for name in MODEL_NAMES:
        model=load_model(MODEL_PATH,name).to(device='cuda',memory_format=model_memory_format(name))
        metrics,prob=validate(model,vb,torch.device('cuda'));actual[name]=prob
        champion['models'][name]['validation']=metrics
        np.save(run_dir/f'{name}_champion_probabilities.npy',prob);del model
    selection=select(actual,valid.labels,[*allowed,STUDENT])
    success=selection['validation']['auroc']>baseline_auc+args.min_delta
    champion['inference_selection']=selection;champion['training_complete']=True
    champion['source_checkpoint_sha256']=SOURCE_SHA;champion['student_epochs_completed']=len(history)
    champion['student_planned_epochs']=args.max_epochs;champion['stopped_after_first_improvement']=success
    atomic_save(champion,MODEL_PATH)
    summary={'training_complete':True,'planned_training_complete':len(history)==args.max_epochs,
        'stopped_after_first_improvement':success,'stopped_for_budget':budget_stop,
        'training_scope':'inserted residual blocks and classifier only','completed':{STUDENT:{'epochs':len(history),'train_samples_per_epoch':len(train)}},
        'selection':selection,'baseline':source['inference_selection'],'model_sha256':digest(MODEL_PATH),
        'source_checkpoint_sha256':SOURCE_SHA,'independent_fp32_reload_validation_passed':True}
    for name in ['TRAINING_COMPLETE.json','POST_SELECTION_COMPLETE.json']:
        (run_dir/name).write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print('TRAINING_COMPLETE '+json.dumps(summary),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-name',default='20261025_deeper_rank')
    p.add_argument('--max-epochs',type=int,default=3);p.add_argument('--learning-rate',type=float,default=3e-5)
    p.add_argument('--head-lr',type=float,default=1e-5);p.add_argument('--min-delta',type=float,default=1e-12)
    p.add_argument('--batch-size',type=int,default=32);p.add_argument('--workers',type=int,default=4)
    p.add_argument('--seed',type=int,default=2026102509);p.add_argument('--resume',action='store_true');run(p.parse_args())
