"""Isolated continuation of GitHub a64b008; official train/valid only.

Resume at a completed epoch with --resume. Never rewrites the legacy submission.
"""
import argparse, copy, hashlib, json, math, random, sys, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

RUN = Path(__file__).resolve().parent
PROJECT = RUN.parents[1]
sys.path.insert(0, str(RUN/'source/submission2/code'))
import utils as u
import train as upstream
from model import MODEL_NAMES, create_model

SOURCE_SHA='2b5a15db0246ee0858e06708105d6c9a36f5b1f028439b2925cb3cabd6ad865e'
RECIPE=u.RECIPE+'_official_only_20261028'

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()

def write(name,obj):
    p=RUN/name;p.parent.mkdir(parents=True,exist_ok=True)
    q=p.with_suffix('.tmp');q.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8');q.replace(p)

def save(obj,p):
    q=p.with_suffix('.tmp');torch.save(obj,q);q.replace(p)

class Cached(Dataset):
    def __init__(self,split,training=False):
        self.split=split;self.array=None
        self.info=json.loads((PROJECT/'work/cache384'/f'{split}.json').read_text())
        self.labels=self.info['labels']
        self.transform=u.FundusDataset([],training).transform
    def __len__(self):return len(self.labels)
    def __getitem__(self,i):
        if self.array is None:self.array=np.load(PROJECT/'work/cache384'/f'{self.split}.npy',mmap_mode='r')
        return self.transform(Image.fromarray(self.array[i])),torch.tensor(self.labels[i],dtype=torch.float32)

def make_loader(ds,training):
    options=dict(num_workers=4,persistent_workers=True,pin_memory=False)
    if training:return DataLoader(ds,batch_sampler=u.BalancedBatchSampler(ds.labels,batch_size=16),**options)
    return DataLoader(ds,batch_size=16,shuffle=False,**options)

@torch.inference_mode()
def validate(model,batches):
    model.eval();scores=[];labels=[];t=time.time()
    for i,(x,y) in enumerate(batches,1):
        scores.extend(torch.sigmoid(model(x.to('cuda',non_blocking=False)).flatten()).cpu().tolist());labels.extend(y.tolist())
        if i==1 or i%100==0 or i==len(batches):print(f'VALID {i}/{len(batches)} seconds={time.time()-t:.1f}',flush=True)
    return u.metrics(labels,scores),np.array(scores)

def selection(vectors,labels):
    vectors={**vectors,'ensemble':sum(vectors.values())/len(vectors)}
    results={n:u.metrics(labels,v) for n,v in vectors.items()}
    name=max(results,key=lambda n:results[n]['auroc'])
    return {'name':name,'validation':results[name],'candidates':results}

def audit():
    report={}
    for split in ('train','valid'):
        records=u.read_records(split);ds=Cached(split)
        assert [p.name for p,l in records]==ds.info['names']
        assert [l for p,l in records]==ds.labels
        with ThreadPoolExecutor(max_workers=8) as pool:hashes=list(pool.map(lambda r:sha(r[0]),records))
        assert hashes==ds.info['sha256'],'Original data differs from cached data'
        arr=np.load(PROJECT/'work/cache384'/f'{split}.npy',mmap_mode='r')
        for j in np.linspace(0,len(records)-1,24,dtype=int):
            with Image.open(records[j][0]) as im:assert np.array_equal(np.array(u.letterbox(im)),arr[j])
        report[split]={'count':len(records),'positive':sum(ds.labels),'all_original_hashes_match':True,'pixel_equality_samples':24}
        print('CACHE_VERIFIED '+split,flush=True)
    write('data_audit.json',report)

def run(args):
    torch.set_num_threads(4);upstream.set_seed(20261028)
    assert torch.cuda.is_available()
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    u.DATA=PROJECT/'glaucoma_first'
    source=RUN/'source/submission/model/best_model.pth';assert sha(source)==SOURCE_SHA
    config={'github_commit':'a64b008f67f9eb3030ffb8e251329ca92e3a4022','source_sha256':SOURCE_SHA,
        'recipe':RECIPE,'gpu':torch.cuda.get_device_name(0),'timm':__import__('timm').__version__,
        'max_epochs_per_branch':args.max_epochs,'stop':'first complete validation improvement over source',
        'train':'official train only; no extra medical data; official valid selection only',
        'precision':'AMP FP16 training; FP32 validation/inference','source_config':upstream.CONFIG,
        'extra_data_enabled':False,'validation_in_training':False,'tta':False,
        'cache':'384 RGB LANCZOS fullframe pixels, hashes and sampled pixels verified',
        'optimizer':'fresh AdamW warm start; exact optimizer/RNG resume at saved epoch boundaries'}
    write('config.json',config)
    audit()
    baseline=u.load_resume_checkpoint(source,MODEL_NAMES,recipe=(u.BASE_RECIPE,RECIPE))
    train_ds=Cached('train',True);valid_ds=Cached('valid');vl=make_loader(valid_ds,False)
    checkpoint_path=RUN/'best_model.pth';progress_path=RUN/'progress.pth'
    if progress_path.exists() and not args.resume:raise RuntimeError('Use --resume')
    if (RUN/'TRAINING_COMPLETE.json').exists():print('Already completed; refusing new training');return
    if not (RUN/'baseline.json').exists():
        vectors={}
        for name in MODEL_NAMES:
            m,_=create_model(name);m.load_state_dict(baseline['states'][name]);m.to('cuda')
            metric,vectors[name]=validate(m,vl)
            assert abs(metric['auroc']-baseline['best_metrics'][name]['auroc'])<.002
            np.save(RUN/f'{name}_baseline.npy',vectors[name]);del m;torch.cuda.empty_cache()
        first=selection(vectors,valid_ds.labels)
        assert abs(first['validation']['auroc']-baseline['selection']['validation']['auroc'])<.002
        write('baseline.json',first);print('BASELINE '+json.dumps(first),flush=True)
        best=copy.deepcopy(baseline);best.update(recipe=RECIPE,selection=first,torch_version=str(torch.__version__))
        save(best,checkpoint_path)
    first=json.loads((RUN/'baseline.json').read_text());reference=first['validation']['auroc']
    best=torch.load(checkpoint_path,weights_only=True,map_location='cpu')
    vectors={n:np.load(RUN/f'{n}_best.npy') if (RUN/f'{n}_best.npy').exists() else np.load(RUN/f'{n}_baseline.npy') for n in MODEL_NAMES}
    previous=torch.load(progress_path,map_location='cpu',weights_only=True) if progress_path.exists() else None
    history=json.loads((RUN/'history.json').read_text()) if (RUN/'history.json').exists() else []
    success=False
    for branch,name in enumerate(MODEL_NAMES):
        if previous and branch<previous['branch']:continue
        m,_=create_model(name);m.load_state_dict(baseline['states'][name]);m.to('cuda')
        cfg=upstream.CONFIG[name];ids={id(p) for p in m.get_classifier().parameters()}
        opt=torch.optim.AdamW([{'params':[p for p in m.parameters() if id(p) not in ids],'lr':cfg['backbone_lr']},
            {'params':[p for p in m.parameters() if id(p) in ids],'lr':cfg['head_lr']}],weight_decay=cfg['weight_decay'])
        scaler=torch.amp.GradScaler('cuda');start=0
        if previous and branch==previous['branch']:
            m.load_state_dict(previous['model']);opt.load_state_dict(previous['optimizer']);scaler.load_state_dict(previous['scaler'])
            torch.set_rng_state(previous['torch_rng']);torch.cuda.set_rng_state_all(previous['cuda_rng'])
            random.setstate(previous['python_rng']);ns=previous['numpy_rng'];np.random.set_state((ns[0],np.array(ns[1],dtype=np.uint32),ns[2],ns[3],ns[4]))
            start=previous['epoch']
        tl=make_loader(train_ds,True)
        for epoch in range(start,args.max_epochs):
            t=time.time();tl.batch_sampler.set_epoch(epoch+1)
            print(f'TRAIN_START {name} epoch={epoch+1}',flush=True)
            loss=upstream.train_epoch(m,tl,opt,scaler,torch.tensor(math.sqrt(15),device='cuda'),torch.device('cuda'),epoch+1,name,.1)
            metric,p=validate(m,vl);trial={**vectors,name:p};chosen=selection(trial,valid_ds.labels)
            better=chosen['validation']['auroc']>best['selection']['validation']['auroc']
            if better:
                best['states'][name]={k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
                best['best_metrics'][name]=metric;vectors[name]=p;np.save(RUN/f'{name}_best.npy',p)
                best['selection']=chosen;best['training_complete']=True;save(best,checkpoint_path)
            row={'branch':name,'epoch':epoch+1,'loss':loss,'seconds':time.time()-t,'validation':metric,'selection':chosen,
                'source_reference':reference,'global_improved':better}
            history.append(row);write('history.json',history);print('EPOCH_RESULT '+json.dumps(row),flush=True)
            # Keep NumPy RNG data as basic tensors/lists for weights_only-safe resume.
            ns=np.random.get_state();nr=(ns[0],ns[1].tolist(),ns[2],ns[3],ns[4])
            save({'branch':branch,'epoch':epoch+1,'model':m.state_dict(),'optimizer':opt.state_dict(),'scaler':scaler.state_dict(),
                'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'python_rng':random.getstate(),'numpy_rng':nr},progress_path)
            if chosen['validation']['auroc']>reference+1e-12:
                success=True;print('FIRST_IMPROVEMENT_STOP',flush=True);break
            if (RUN/'STOP_AFTER_EPOCH').exists():write('STOPPED.json',{'reason':'requested boundary stop','history':history});return
        del m,opt,scaler,tl;torch.cuda.empty_cache()
        if success:break
    # Independently reload both full networks and reproduce the selected score.
    actual={}
    for name in MODEL_NAMES:
        m,_=create_model(name);m.load_state_dict(best['states'][name]);m.to('cuda')
        _,actual[name]=validate(m,vl);del m;torch.cuda.empty_cache()
    final=selection(actual,valid_ds.labels)
    assert abs(final['validation']['auroc']-best['selection']['validation']['auroc'])<1e-10
    best['selection']=final;best['source_checkpoint_sha256']=SOURCE_SHA;best['training_complete']=True
    save(best,checkpoint_path)
    report={'status':'stopped_after_improvement' if success else 'bounded_training_complete_no_improvement',
        'baseline':first,'final':final,'gain':final['validation']['auroc']-reference,'history':history,
        'model_sha256':sha(checkpoint_path),'independent_full_network_reload_passed':True}
    write('TRAINING_COMPLETE.json',report);print('TRAINING_COMPLETE '+json.dumps(report),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--resume',action='store_true');p.add_argument('--max-epochs',type=int,default=3)
    run(p.parse_args())
