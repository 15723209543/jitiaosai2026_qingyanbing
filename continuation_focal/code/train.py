"""GPU classification-head fine-tuning on immutable FP32 source-encoder features."""
import argparse,copy,hashlib,json,math,random,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from model import MODEL_NAMES
from prepare import prepare
from training_core import CachedDataset,loader,validate,atomic_save
from utils import PROJECT_DIR,MODEL_PATH,RECIPE,load_model,model_memory_format,calculate_metrics

SOURCE_SHA='f7d35a6bb77819750d4b59fdeac0f273b775578ed8ec1bf39e6bafa525a7f6f5'
def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def candidates(probabilities,labels):
    metrics={n:calculate_metrics(labels,probabilities[n]) for n in MODEL_NAMES}
    metrics['fixed_mean_ensemble']=calculate_metrics(labels,np.mean(list(probabilities.values()),axis=0))
    key,names=max([('fixed_mean_ensemble',list(MODEL_NAMES)),*[(n,[n]) for n in MODEL_NAMES]],key=lambda p:metrics[p[0]]['auroc'])
    return {'model_names':names,'metric':'official_validation_auroc','candidate_metrics':metrics,
        'selected_candidate':key,'validation':metrics[key],'threshold':.5,'tta':False,
        'inference_precision':'FP32','rule':'validation AUROC; single branches and fixed equal-mean snapshots; no test data'}
def head_of(model,name):
    return model.network.fc if name=='resnet18' else model.network.classifier

@torch.inference_mode()
def features(model,name,split,excluded,run_dir,workers):
    original_run=PROJECT_DIR/'runs/20261025_frozen_head'
    original_model=original_run/'baseline_snapshot/model/best_model.pth'
    original_sha=digest(original_model)
    original=torch.load(original_model,map_location='cpu',weights_only=True)['models'][name]['state_dict']
    assert all(torch.equal(v.cpu(),original[k]) for k,v in model.state_dict().items()
        if not k.startswith(('network.fc.','network.classifier.'))),'Reused encoder changed'
    path=original_run/'features'/f'{name}_{split}.npy'
    meta=json.loads(path.with_suffix('.json').read_text())
    assert meta['encoder_source_sha256']==original_sha and meta['precision']=='FP32'
    assert meta['image_size']==256 and not meta['training_augmentation']
    assert meta['excluded_train_indices']==(list(excluded) if split=='train' else [])
    assert digest(path)==meta['sha256']
    values=np.load(path)
    assert list(values.shape)==meta['shape'] and np.isfinite(values).all()
    dest=run_dir/'features';dest.mkdir(exist_ok=True)
    (dest/f'{name}_{split}.json').write_text(json.dumps({**meta,'reused_from':str(path),
        'current_checkpoint_sha256':SOURCE_SHA,'encoder_equality_verified':True},indent=2),encoding='utf-8')
    print(f'REUSE_FEATURES {name} {split} {values.shape} verified',flush=True)
    return values

def focal_loss_with_diagnostics(logits,labels,criterion):
    probability=torch.softmax(logits,1).gather(1,labels[:,None]).squeeze(1)
    ce=torch.nn.functional.cross_entropy(logits,labels,weight=criterion.weight,reduction='none')
    loss=((1-probability)*ce).sum()/criterion.weight[labels].sum()
    return loss,loss.detach(),criterion(logits,labels)

@torch.inference_mode()
def head_validation(head,x,y):
    head.eval();values=[]
    for start in range(0,len(x),512):values.extend(torch.softmax(head(x[start:start+512]),1)[:,1].cpu().tolist())
    return calculate_metrics(y,values),values
def entry(model,metrics,epoch):
    return {'state_dict':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
        'model_config':dict(model.model_config),'validation':metrics,'epoch':epoch,
        'training_scope':'classifier head only; encoder and BN frozen','selection_metric':'validation_auroc'}

def run(args):
    torch.set_num_threads(4);torch.manual_seed(args.seed);np.random.seed(args.seed);random.seed(args.seed)
    if not torch.cuda.is_available():raise RuntimeError('GPU required')
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    run_dir=PROJECT_DIR/'runs'/args.run_name;baseline=run_dir/'baseline_snapshot/model/best_model.pth'
    assert digest(baseline)==SOURCE_SHA
    config={**vars(args),'recipe':RECIPE,'source_checkpoint_sha256':SOURCE_SHA,
        'initialization':'published competition best weights; fresh AdamW for heads',
        'precision':'FP32 feature extraction, GPU head optimization and inference',
        'training_scope':'classifier only; source encoder and BN buffers unchanged',
        'objective':'class-weighted focal loss gamma1','feature_augmentation':False,'dropout':.2,'weight_decay':.01,
        'validation_use':'selection only','test_use':'fixed final inference only',
        'gpu':torch.cuda.get_device_name(0),'torch':str(torch.__version__),
        'source_files_sha256':{p.name:digest(p) for p in Path(__file__).parent.glob('*.py')}}
    config_path=run_dir/'config.json'
    if config_path.exists():
        if not args.resume:raise FileExistsError('Use --resume for existing run')
        previous=json.loads(config_path.read_text())
        for k in ['recipe','source_checkpoint_sha256','max_epochs','min_epochs','patience','learning_rate','min_lr','batch_size','seed']:
            assert previous[k]==config[k],k
    config_path.write_text(json.dumps(config,indent=2),encoding='utf-8')
    audit=prepare();excluded=audit['train_indices_excluded_exact_validation_duplicates']
    train_ds=CachedDataset('train',False,excluded);valid_ds=CachedDataset('valid')
    y=torch.tensor(train_ds.labels[train_ds.indices],device='cuda')
    validation_loader=loader(valid_ds,64,args.workers);branch_path=run_dir/'branch_best.pth'
    if not branch_path.exists():
        bundle=torch.load(baseline,map_location='cpu',weights_only=True)
        for n in MODEL_NAMES:bundle['models'][n]['model_config']['recipe']=RECIPE
        bundle['training_complete']=False;atomic_save(bundle,branch_path);probabilities={}
        for n in MODEL_NAMES:
            model=load_model(branch_path,n).to(device='cuda',memory_format=model_memory_format(n))
            metrics,prob=validate(model,validation_loader,torch.device('cuda'));probabilities[n]=prob
            bundle['models'][n]['validation']=metrics
            for tag in ['baseline','best','champion']:np.save(run_dir/f'{n}_{tag}_probabilities.npy',prob)
            del model
        selection=candidates(probabilities,valid_ds.labels);bundle['inference_selection']=selection
        atomic_save(bundle,branch_path);atomic_save(bundle,MODEL_PATH)
        (run_dir/'baseline_metrics.json').write_text(json.dumps(selection,indent=2),encoding='utf-8')
        print('BASELINE '+json.dumps(selection),flush=True)
    criterion=nn.CrossEntropyLoss(weight=torch.tensor([1.,math.sqrt((len(y)-int(y.sum()))/int(y.sum()))],device='cuda'))
    completed={n:{'additional_epochs':0,'stopped_early':False,'scope':'unchanged legacy branch'} for n in MODEL_NAMES}
    stop_success=False
    reference=json.loads((run_dir/'baseline_metrics.json').read_text())['validation']['auroc']
    for name in MODEL_NAMES:
        source=load_model(branch_path,name)
        train_features=features(source,name,'train',excluded,run_dir,args.workers)
        valid_features=features(source,name,'valid',(),run_dir,args.workers)
        x=torch.from_numpy(train_features).to('cuda');vx=torch.from_numpy(valid_features).to('cuda')
        for parameter in source.parameters():parameter.requires_grad_(False)
        head=head_of(source,name).to('cuda')
        for parameter in head.parameters():parameter.requires_grad_(True)
        optimizer=torch.optim.AdamW(head.parameters(),lr=args.learning_rate,weight_decay=.01)
        history=[];start_epoch=0;no_improvement=0;stopped=False
        last_path=run_dir/f'{name}_last.pth'
        if last_path.exists():
            state=torch.load(last_path,map_location='cpu',weights_only=False)
            head.load_state_dict(state['head']);optimizer.load_state_dict(state['optimizer'])
            start_epoch=state['epoch'];history=state['history'];no_improvement=state['no_improvement'];stopped=state['stopped_early']
            torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state_all(state['cuda_rng'])
        else:
            torch.manual_seed(args.seed+MODEL_NAMES.index(name));torch.cuda.manual_seed_all(args.seed+MODEL_NAMES.index(name))
        frozen={k:v.clone() for k,v in source.state_dict().items() if not k.startswith(('network.fc.','network.classifier.'))}
        print(f'TRAIN_HEAD {name} start={start_epoch} samples={len(x)} feature_dim={x.shape[1]}',flush=True)
        for epoch in range(start_epoch,args.max_epochs):
            if stopped:break
            started=time.perf_counter();lr=args.min_lr+(args.learning_rate-args.min_lr)*(1+math.cos(math.pi*epoch/max(1,args.max_epochs-1)))/2
            for group in optimizer.param_groups:group['lr']=lr
            head.train();indices=torch.randperm(len(x),device='cuda');loss_sum=0.;rank_sum=0.;ce_sum=0.;count=0
            for batch in indices.split(args.batch_size):
                optimizer.zero_grad(set_to_none=True);loss,rank_loss,ce_loss=focal_loss_with_diagnostics(head(x[batch]),y[batch],criterion)
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite head loss')
                loss.backward();nn.utils.clip_grad_norm_(head.parameters(),5.);optimizer.step()
                loss_sum+=float(loss.detach())*len(batch);rank_sum+=float(rank_loss.detach())*len(batch);ce_sum+=float(ce_loss.detach())*len(batch);count+=len(batch)
            assert count==len(train_ds)
            metrics,prob=head_validation(head,vx,valid_ds.labels)
            best=torch.load(branch_path,map_location='cpu',weights_only=True);before_branch=best['models'][name]['validation']['auroc']
            other=next(n for n in MODEL_NAMES if n!=name);pair={name:prob,other:np.load(run_dir/f'{other}_best_probabilities.npy')}
            selection=candidates(pair,valid_ds.labels);champion=torch.load(MODEL_PATH,map_location='cpu',weights_only=True)
            before_global=champion['inference_selection']['validation']['auroc'];current_entry=entry(source,metrics,epoch+1)
            assert all(torch.equal(v,source.state_dict()[k].cpu()) for k,v in frozen.items()),'Encoder changed'
            if selection['validation']['auroc']>before_global:
                best['models'][name]=current_entry;best['inference_selection']=selection;best['training_complete']=False
                atomic_save(best,MODEL_PATH)
                print('CHAMPION '+json.dumps({'model':name,'epoch':epoch+1,'selection':selection}),flush=True)
            best=torch.load(branch_path,map_location='cpu',weights_only=True)
            if metrics['auroc']>before_branch:
                best['models'][name]=current_entry;atomic_save(best,branch_path)
                np.save(run_dir/f'{name}_best_probabilities.npy',prob)
                atomic_save({'head':head.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch+1,
                    'encoder_source_sha256':SOURCE_SHA},run_dir/f'{name}_best_head_optimizer.pth')
            meaningful=metrics['auroc']>before_branch+args.min_delta or selection['validation']['auroc']>before_global+args.min_delta
            no_improvement=0 if meaningful else no_improvement+1;stopped=epoch+1>=args.min_epochs and no_improvement>=args.patience
            row={'epoch':epoch+1,'train_samples':count,'train_loss':loss_sum/count,'focal_loss':rank_sum/count,'ce_loss':ce_sum/count,'learning_rate':lr,
                'epoch_seconds':time.perf_counter()-started,**metrics,'best_branch_auroc':max(before_branch,metrics['auroc']),
                'best_global_auroc':max(before_global,selection['validation']['auroc']),'no_improvement':no_improvement,
                'stopped_early':stopped,'encoder_unchanged':True}
            history.append(row);pd.DataFrame(history).to_csv(run_dir/f'{name}_history.csv',index=False)
            atomic_save({'head':head.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch+1,'history':history,
                'no_improvement':no_improvement,'stopped_early':stopped,'torch_rng':torch.get_rng_state(),
                'cuda_rng':torch.cuda.get_rng_state_all()},last_path)
            print('EPOCH_RESULT '+json.dumps({'model':name,**row}),flush=True)
            if selection['validation']['auroc']>reference+1e-12:
                stop_success=True
                print('FIRST_IMPROVEMENT_STOP',flush=True)
                break
            if (run_dir/'STOP_AFTER_EPOCH').exists():
                (run_dir/'BUDGET_STOPPED.json').write_text(json.dumps({'model':name,'epoch':epoch+1,'complete':False}),encoding='utf-8')
                print('Stopped at checkpoint boundary for quota limit',flush=True)
                return
        completed[name]={'additional_epochs':len(history),'stopped_early':stopped,'scope':'classification head only'}
        del source,head,optimizer,x,vx
        torch.cuda.empty_cache()
        if stop_success:break
    champion=torch.load(MODEL_PATH,map_location='cpu',weights_only=True);actual={}
    for name in MODEL_NAMES:
        model=load_model(MODEL_PATH,name).to(device='cuda',memory_format=model_memory_format(name))
        metrics,prob=validate(model,validation_loader,torch.device('cuda'));champion['models'][name]['validation']=metrics
        actual[name]=prob;np.save(run_dir/f'{name}_champion_probabilities.npy',prob);del model
        model=load_model(branch_path,name).to(device='cuda',memory_format=model_memory_format(name))
        metrics,prob=validate(model,validation_loader,torch.device('cuda'));np.save(run_dir/f'{name}_best_probabilities.npy',prob);del model
    champion['inference_selection']=candidates(actual,valid_ds.labels);champion['training_complete']=True
    champion['additional_head_epochs_completed']={n:v['additional_epochs'] for n,v in completed.items()}
    champion['initialization_source_sha256']=SOURCE_SHA;atomic_save(champion,MODEL_PATH)
    summary={'completed':completed,'selection':champion['inference_selection'],
        'baseline':json.loads((run_dir/'baseline_metrics.json').read_text()),'model_sha256':digest(MODEL_PATH),
        'source_checkpoint_sha256':SOURCE_SHA,'training_complete':True,'encoder_unchanged':True,'recipe':RECIPE}
    (run_dir/'TRAINING_COMPLETE.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    (run_dir/'inference_selection.json').write_text(json.dumps(champion['inference_selection'],indent=2),encoding='utf-8')
    print('TRAINING_COMPLETE '+json.dumps(summary),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-name',default='20261025_focal_head')
    parser.add_argument('--max-epochs',type=int,default=12);parser.add_argument('--min-epochs',type=int,default=1)
    parser.add_argument('--patience',type=int,default=5);parser.add_argument('--min-delta',type=float,default=1e-5)
    parser.add_argument('--learning-rate',type=float,default=1e-4);parser.add_argument('--min-lr',type=float,default=1e-5)
    parser.add_argument('--batch-size',type=int,default=512);parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--seed',type=int,default=2026102510);parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    assert 1<=args.min_epochs<=args.max_epochs and args.patience>0 and 0<args.min_lr<=args.learning_rate
    run(args)
