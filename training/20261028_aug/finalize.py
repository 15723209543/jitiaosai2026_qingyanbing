"""Verify an enhanced-data run and promote only a strict validation improvement.

Dataset images, YOLO labels and per-image manifests are never copied to the repository.
"""
import argparse,csv,hashlib,importlib.util,json,shutil,sys,time
from pathlib import Path
import torch
from torch.utils.data import DataLoader

P=Path(__file__).resolve().parents[2]
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()

def finalize(run):
    run=run.resolve()
    if not run.is_relative_to(P/'runs'):raise ValueError('Run outside project/runs')
    report=json.loads((run/'TRAINING_COMPLETE.json').read_text(encoding='utf-8'))
    if report['gain']<=0:
        # Failed experiments retain the original weights and their actual recipe.
        source_path=run/'source/submission/model/best_model.pth'
        with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
            source=torch.load(source_path,map_location='cpu',weights_only=True)
        retained=torch.load(run/'best_model.pth',map_location='cpu',weights_only=True)
        assert set(source['states'])==set(retained['states'])
        for name,state in source['states'].items():
            assert all(torch.equal(v,retained['states'][name][k]) for k,v in state.items())
        retained['attempted_recipe']=retained['recipe'];retained['recipe']=source['recipe']
        retained['weights_retained_from_source']=True
        temp=run/'retained_model.tmp';torch.save(retained,temp);temp.replace(run/'best_model.pth')
        report['model_sha256']=sha(run/'best_model.pth');report['weights_retained_from_source']=True
        report['official_model_sha256']=sha(P/'submission/model/best_model.pth')
        report['official_csv_sha256']=sha(P/'submission/submission.csv')
        assert report['official_model_sha256']==sha(source_path)
        (run/'TRAINING_COMPLETE.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        print('No validation improvement: official model and CSV preserved.',flush=True);return
    cfg=json.loads((run/'config.json').read_text(encoding='utf-8'))
    assert sha(P/'submission/model/best_model.pth')==cfg['source_sha256'],'Official model changed concurrently'
    assert sha(run/'best_model.pth')==report['model_sha256']
    bundle=torch.load(run/'best_model.pth',map_location='cpu',weights_only=True)
    candidate=run/'candidate';(candidate/'code').mkdir(parents=True,exist_ok=True);(candidate/'model').mkdir(exist_ok=True)
    for f in (P/'submission/code').glob('*'):
        if not f.is_file():continue
        text=f.read_text(encoding='utf-8')
        if f.name=='utils.py':
            import re
            text,n=re.subn(r'^RECIPE = .+$','RECIPE = '+repr(bundle['recipe']),text,flags=re.M);assert n==1
        (candidate/'code'/f.name).write_text(text,encoding='utf-8')
    shutil.copy2(run/'best_model.pth',candidate/'model/best_model.pth')
    shutil.copy2(P/'submission/cost.py',candidate/'cost.py')
    sys.path.insert(0,str(candidate/'code'))
    import utils as u
    import predict
    u.DATA=P/'glaucoma_first'
    models=predict.load_inference_models()
    records=u.read_records('test');scores=[];start=time.time()
    loader=DataLoader(u.FundusDataset(records,False),batch_size=1,num_workers=4,pin_memory=False)
    for i,images in enumerate(loader,1):
        scores.extend(predict.score_batch(images,models))
        if i==1 or i%1000==0 or i==len(records):print(f'PREDICT {i}/{len(records)} seconds={time.time()-start:.1f}',flush=True)
    csv_path=u.write_submission(records,scores,candidate/'submission.csv')
    rows=list(csv.DictReader(csv_path.open(encoding='utf-8')))
    with (u.DATA/'test.csv').open(encoding='utf-8-sig',newline='') as f:expected_names=[r['Image_name'] for r in csv.DictReader(f)]
    assert [r['Image_name'] for r in rows]==expected_names and len(rows)==5855
    spec=importlib.util.spec_from_file_location('candidate_cost',candidate/'cost.py')
    cost=importlib.util.module_from_spec(spec);spec.loader.exec_module(cost)
    cost.run_round([p for p,l in records[:5]],models,run/'warmup.csv')
    expected={r['Image_name'].casefold():float(r['Probability']) for r in rows}
    durations=[];mismatches=0
    for i in range(3):
        times,values=cost.run_round([p for p,l in records[5:105]],models,run/f'cost_round{i+1}.csv')
        durations.extend(times);mismatches+=sum(v!=expected[n.casefold()] for n,v in values.items())
    assert mismatches==0
    verification={'model_sha256':sha(candidate/'model/best_model.pth'),'csv_sha256':sha(csv_path),'rows':len(rows),
        'official_names_and_order':True,'csv_inference_mismatches':mismatches,'timing_warmup':5,'timing_images':100,'timing_repeats':3,
        'average_seconds_per_image':sum(durations)/len(durations),'official_speed_limit_passed':sum(durations)/len(durations)<.1,
        'validation':report['final'],'source_validation':report['baseline'],'dataset_published':False}
    (run/'verification.json').write_text(json.dumps(verification,indent=2),encoding='utf-8')
    backup=run/'previous_official';backup.mkdir(exist_ok=True)
    for relative in ['model/best_model.pth','submission.csv','code/utils.py']:
        dest=backup/relative;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(P/'submission'/relative,dest)
    for relative in ['model/best_model.pth','submission.csv','code/utils.py']:
        shutil.copy2(candidate/relative,P/'submission'/relative)
    (run/'PROMOTED.json').write_text(json.dumps(verification,indent=2),encoding='utf-8')
    for name in ['verification.json','PROMOTED.json']:shutil.copy2(run/name,P/'training/20261028_aug'/name)
    print('PROMOTED '+json.dumps(verification),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('run',type=Path)
    finalize(parser.parse_args().run)
