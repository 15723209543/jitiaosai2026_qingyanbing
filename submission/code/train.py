"""Continue the validated checkpoint with small LR; retain the best whole inference snapshot."""
import argparse
import copy
import hashlib
import json
import math
import random
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from model import MODEL_NAMES
from prepare import prepare
from training_core import CachedDataset, loader, validate, atomic_save
from utils import PROJECT_DIR, MODEL_PATH, RECIPE, calculate_metrics, load_model, model_memory_format, save_model

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def candidates(probabilities, labels):
    metrics = {n: calculate_metrics(labels, probabilities[n]) for n in MODEL_NAMES}
    metrics['fixed_mean_ensemble'] = calculate_metrics(labels, np.mean([probabilities[n] for n in MODEL_NAMES],axis=0))
    options = [('fixed_mean_ensemble', list(MODEL_NAMES)), *[(n,[n]) for n in MODEL_NAMES]]
    key,names = max(options,key=lambda item: metrics[item[0]]['auroc'])
    return {'model_names':names,'metric':'official_validation_auroc','candidate_metrics':metrics,
            'selected_candidate':key,'validation':metrics[key],
            'rule':'best validation AUROC across observed single branches and fixed equal-mean snapshots; no test data',
            'threshold':0.5,'tta':False,'inference_precision':'FP32'}

def state_entry(model, metrics, epoch):
    return {'state_dict': {k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
            'model_config':dict(model.model_config),'validation':metrics,'epoch':epoch,
            'prior_local_training_epochs':5,'selection_metric':'validation_auroc'}

def run(args):
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU is required')
    device=torch.device('cuda')
    torch.backends.cudnn.benchmark=True
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_tf32=False
    source_run=PROJECT_DIR/'runs'/'20261024_upstream_dual'
    run_dir=PROJECT_DIR/'runs'/args.run_name
    run_dir.mkdir(parents=True,exist_ok=True)
    baseline_path=run_dir/'baseline_snapshot'/'model'/'best_model.pth'
    if not baseline_path.exists():
        raise FileNotFoundError('Immutable baseline snapshot is required')
    source_sha=digest(baseline_path)
    if source_sha!='134650d23ce1f2c48bdf2e9abf612faff61b3ccf6cd0b39530cba8c1963b8eaf':
        raise ValueError('Continuation source checkpoint differs from the verified previous round')
    config={**vars(args),'recipe':RECIPE,'initialization_sha256':source_sha,'initialization':'previous verified best plus matching optimizer/scaler/RNG states',
            'prior_local_epochs_each':5,'schedule':'small-LR cosine, no restart warmup','precision':'AMP FP16 training, FP32 validation/inference',
            'validation_use':'checkpoint/snapshot selection only','test_use':'final fixed inference only',
            'source_run':str(source_run),'gpu':torch.cuda.get_device_name(0),'torch':str(torch.__version__),
            'source_files_sha256':{p.name:digest(p) for p in sorted(Path(__file__).parent.glob('*.py'))}}
    config_path=run_dir/'config.json'
    if config_path.exists():
        if not args.resume:
            raise FileExistsError('Use --resume for an existing continuation run')
        previous=json.loads(config_path.read_text(encoding='utf-8'))
        for k in ['recipe','initialization_sha256','max_epochs','min_epochs','patience','learning_rate','min_lr','batch_size','seed']:
            if previous[k]!=config[k]:
                raise ValueError('Resume configuration changed: '+k)
    config_path.write_text(json.dumps(config,indent=2),encoding='utf-8')
    audit=prepare()
    train_ds=CachedDataset('train',True,audit['train_indices_excluded_exact_validation_duplicates'])
    valid_ds=CachedDataset('valid')
    valid_loader=loader(valid_ds,64,args.workers)
    branch_path=run_dir/'branch_best.pth'
    if not branch_path.exists():
        initial=torch.load(baseline_path,map_location='cpu',weights_only=True)
        initial.pop('inference_selection',None)
        initial['training_complete']=False
        for n in MODEL_NAMES:
            initial['models'][n]['model_config']['recipe']=RECIPE
        atomic_save(initial,branch_path)
        probabilities={}
        for n in MODEL_NAMES:
            model=load_model(branch_path,n).to(device=device,memory_format=model_memory_format(n))
            metrics,prob=validate(model,valid_loader,device)
            probabilities[n]=prob
            initial['models'][n]['validation']=metrics
            np.save(run_dir/f'{n}_best_probabilities.npy',prob)
            np.save(run_dir/f'{n}_baseline_probabilities.npy',prob)
            del model
        atomic_save(initial,branch_path)
        selection=candidates(probabilities,valid_ds.labels)
        initial['inference_selection']=selection
        atomic_save(initial,MODEL_PATH)
        for n in MODEL_NAMES:
            np.save(run_dir/f'{n}_champion_probabilities.npy',probabilities[n])
        (run_dir/'baseline_metrics.json').write_text(json.dumps(selection,indent=2),encoding='utf-8')
        print('BASELINE '+json.dumps(selection),flush=True)
    completed={}
    for name in MODEL_NAMES:
        last_path=run_dir/f'{name}_last.pth'
        best_bundle=torch.load(branch_path,map_location='cpu',weights_only=True)
        model=load_model(branch_path,name).to(device=device,memory_format=model_memory_format(name))
        optimizer=torch.optim.AdamW(model.parameters(),lr=args.learning_rate,weight_decay=1e-4)
        scaler=torch.amp.GradScaler('cuda',init_scale=1024.)
        history=[]
        epoch_start=0
        no_improvement=0
        stopped=False
        if last_path.exists():
            state=torch.load(last_path,map_location='cpu',weights_only=False)
            model.load_state_dict(state['model'])
            epoch_start=state['epoch']
            history=state['history']
            no_improvement=state['no_improvement']
            stopped=state.get('stopped_early',False)
        else:
            state=torch.load(source_run/f'{name}_last.pth',map_location='cpu',weights_only=False)
            expected=torch.load(baseline_path,map_location='cpu',weights_only=True)['models'][name]['state_dict']
            if not all(torch.equal(state['model'][k],expected[k]) for k in expected):
                raise ValueError('Previous optimizer does not match the best source weights: '+name)
            history=[{'epoch':0,'local_total_epoch':5,**best_bundle['models'][name]['validation']}]
        optimizer.load_state_dict(state['optimizer'])
        scaler.load_state_dict(state['scaler'])
        torch.set_rng_state(state['torch_rng'])
        torch.cuda.set_rng_state_all(state['cuda_rng'])
        random.setstate(state['python_rng'])
        np.random.set_state(state['numpy_rng'])
        labels=train_ds.labels[train_ds.indices]
        positives=int(labels.sum())
        criterion=nn.CrossEntropyLoss(weight=torch.tensor([1.,math.sqrt((len(labels)-positives)/positives)],device=device))
        print(f'TRAIN {name} start={epoch_start} max_additional_epochs={args.max_epochs} samples={len(train_ds)}',flush=True)
        for epoch in range(epoch_start,args.max_epochs):
            if stopped:
                break
            lr=args.min_lr+(args.learning_rate-args.min_lr)*(1+math.cos(math.pi*epoch/max(1,args.max_epochs-1)))/2
            for group in optimizer.param_groups:
                group['lr']=lr
            batches=loader(train_ds,args.batch_size,args.workers,True,args.seed+5+epoch)
            model.train()
            total_loss,count,started=0.,0,time.perf_counter()
            for step,(x,y) in enumerate(batches,1):
                x=x.to(device,non_blocking=True,memory_format=model_memory_format(name))
                y=y.to(device,non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.float16):
                    loss=criterion(model(x),y)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite loss')
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(),5.)
                scaler.step(optimizer)
                scaler.update()
                count+=len(y)
                total_loss+=float(loss.detach())*len(y)
                if step==1 or step%100==0 or step==len(batches):
                    elapsed=time.perf_counter()-started
                    print(f'{name} additional_epoch={epoch+1}/{args.max_epochs} batch={step}/{len(batches)} loss={total_loss/count:.5f} elapsed={elapsed:.1f}s',flush=True)
            del batches
            if count!=len(train_ds):
                raise RuntimeError('Incomplete training epoch')
            metrics,prob=validate(model,valid_loader,device)
            before_branch=best_bundle['models'][name]['validation']['auroc']
            branch_improved=metrics['auroc']>before_branch
            other=next(n for n in MODEL_NAMES if n!=name)
            pair={name:prob,other:np.load(run_dir/f'{other}_best_probabilities.npy').tolist()}
            selection=candidates(pair,valid_ds.labels)
            champion=torch.load(MODEL_PATH,map_location='cpu',weights_only=True)
            before_global=champion['inference_selection']['validation']['auroc']
            global_improved=selection['validation']['auroc']>before_global
            if global_improved:
                candidate=copy.deepcopy(best_bundle)
                candidate['models'][name]=state_entry(model,metrics,epoch+1)
                candidate['inference_selection']=selection
                candidate['training_complete']=False
                candidate['initialization_source_sha256']=source_sha
                atomic_save(candidate,MODEL_PATH)
                for n in MODEL_NAMES:
                    np.save(run_dir/f'{n}_champion_probabilities.npy',pair[n])
                print('CHAMPION '+json.dumps({'model':name,'epoch':epoch+1,'selection':selection}),flush=True)
            if branch_improved:
                save_model(model,branch_path,metrics,epoch+1)
                best_bundle=torch.load(branch_path,map_location='cpu',weights_only=True)
                np.save(run_dir/f'{name}_best_probabilities.npy',prob)
            meaningful=(metrics['auroc']>before_branch+args.min_delta or selection['validation']['auroc']>before_global+args.min_delta)
            no_improvement=0 if meaningful else no_improvement+1
            stopped=epoch+1>=args.min_epochs and no_improvement>=args.patience
            entry={'epoch':epoch+1,'local_total_epoch':6+epoch,'train_samples':count,'train_loss':total_loss/count,
                   'learning_rate':lr,'epoch_seconds':time.perf_counter()-started,**metrics,
                   'best_branch_auroc':best_bundle['models'][name]['validation']['auroc'],
                   'best_global_auroc':max(before_global,selection['validation']['auroc']),
                   'no_improvement':no_improvement,'stopped_early':stopped}
            history.append(entry)
            pd.DataFrame(history).to_csv(run_dir/f'{name}_history.csv',index=False)
            atomic_save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict(),
                         'epoch':epoch+1,'history':history,'no_improvement':no_improvement,'stopped_early':stopped,
                         'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),
                         'python_rng':random.getstate(),'numpy_rng':np.random.get_state()},last_path)
            print('EPOCH_RESULT '+json.dumps({'model':name,**entry}),flush=True)
        completed[name]={'additional_epochs':max(row['epoch'] for row in history),'stopped_early':stopped}
        del model,optimizer,scaler
        torch.cuda.empty_cache()
    champion=torch.load(MODEL_PATH,map_location='cpu',weights_only=True)
    actual_probabilities={}
    for name in MODEL_NAMES:
        model=load_model(MODEL_PATH,name).to(device=device,memory_format=model_memory_format(name))
        metrics,prob=validate(model,valid_loader,device)
        actual_probabilities[name]=prob
        champion['models'][name]['validation']=metrics
        np.save(run_dir/f'{name}_champion_probabilities.npy',prob)
        del model
    selection=candidates(actual_probabilities,valid_ds.labels)
    champion['inference_selection']=selection
    champion['training_complete']=True
    champion['epochs_trained_each']={n:5+completed[n]['additional_epochs'] for n in MODEL_NAMES}
    champion['additional_epochs_completed']={n:completed[n]['additional_epochs'] for n in MODEL_NAMES}
    champion['initialization_source_sha256']=source_sha
    atomic_save(champion,MODEL_PATH)
    summary={'completed':completed,'selection':selection,'baseline':json.loads((run_dir/'baseline_metrics.json').read_text()),
             'model_sha256':digest(MODEL_PATH),'source_checkpoint_sha256':source_sha,'training_complete':True}
    (run_dir/'TRAINING_COMPLETE.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    (run_dir/'inference_selection.json').write_text(json.dumps(selection,indent=2),encoding='utf-8')
    print('TRAINING_COMPLETE '+json.dumps(summary),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--max-epochs',type=int,default=8)
    parser.add_argument('--min-epochs',type=int,default=4)
    parser.add_argument('--patience',type=int,default=3)
    parser.add_argument('--min-delta',type=float,default=1e-5)
    parser.add_argument('--learning-rate',type=float,default=3e-5)
    parser.add_argument('--min-lr',type=float,default=1e-6)
    parser.add_argument('--batch-size',type=int,default=64)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--seed',type=int,default=20261024)
    parser.add_argument('--run-name',default='20261024_continue_lr3e5')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    if not 1<=args.min_epochs<=args.max_epochs or args.patience<1 or not 0<args.min_lr<=args.learning_rate:
        parser.error('Invalid continuation schedule')
    run(args)
