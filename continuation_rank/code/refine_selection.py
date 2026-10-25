"""Compare the finite baseline/branch-best/champion fixed-mean combinations, validation only."""
import copy
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np
import torch
from training_core import CachedDataset, loader, validate, atomic_save
from utils import PROJECT_DIR, MODEL_PATH, RECIPE, calculate_metrics, load_model, model_memory_format
from model import MODEL_NAMES

RUN=PROJECT_DIR/'runs/20261025_rank_head'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def run():
    if (RUN/'POST_SELECTION_COMPLETE.json').exists():
        record=json.loads((RUN/'POST_SELECTION_COMPLETE.json').read_text())
        assert sha(MODEL_PATH)==record['model_sha256']
        print('Post-selection already complete',flush=True)
        return
    record=json.loads((RUN/'TRAINING_COMPLETE.json').read_text())
    assert sha(MODEL_PATH)==record['model_sha256']
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_tf32=False
    bundles={
        'baseline':torch.load(RUN/'baseline_snapshot/model/best_model.pth',map_location='cpu',weights_only=True),
        'best':torch.load(RUN/'branch_best.pth',map_location='cpu',weights_only=True),
        'champion':torch.load(MODEL_PATH,map_location='cpu',weights_only=True)}
    dataset=CachedDataset('valid')
    options={}
    for name in MODEL_NAMES:
        options[name]=[]
        for source in bundles:
            probabilities=np.load(RUN/f'{name}_{source}_probabilities.npy')
            if any(np.array_equal(probabilities,p) for _,p in options[name]):
                continue
            options[name].append((source,probabilities))
    trials=[]
    # Always keep the already selected whole snapshot as a candidate.
    winner=copy.deepcopy(bundles['champion'])
    winner_auc=winner['inference_selection']['validation']['auroc']
    winner_sources={n:'champion' for n in MODEL_NAMES}
    winner_prob={n:np.load(RUN/f'{n}_champion_probabilities.npy') for n in MODEL_NAMES}
    for combination in itertools.product(*(options[n] for n in MODEL_NAMES)):
        sources={n:combination[i][0] for i,n in enumerate(MODEL_NAMES)}
        probabilities={n:combination[i][1] for i,n in enumerate(MODEL_NAMES)}
        metrics={n:calculate_metrics(dataset.labels,probabilities[n]) for n in MODEL_NAMES}
        metrics['fixed_mean_ensemble']=calculate_metrics(dataset.labels,np.mean(list(probabilities.values()),axis=0))
        for key,names in [('fixed_mean_ensemble',list(MODEL_NAMES)),*[(n,[n]) for n in MODEL_NAMES]]:
            value=metrics[key]['auroc']
            trials.append({'sources':sources,'candidate':key,'auroc':value})
            if value>winner_auc+1e-12:
                winner=copy.deepcopy(bundles['champion'])
                for n in MODEL_NAMES:
                    winner['models'][n]=copy.deepcopy(bundles[sources[n]]['models'][n])
                    winner['models'][n]['model_config']['recipe']=RECIPE
                winner['inference_selection'].update(model_names=names,candidate_metrics=metrics,
                    selected_candidate=key,validation=metrics[key],
                    rule='validation-only finite baseline/branch-best/champion snapshot combinations; fixed equal mean; no test data')
                winner_sources=sources
                winner_prob=probabilities
                winner_auc=value
    changed=winner_auc>record['selection']['validation']['auroc']+1e-12
    if changed:
        atomic_save(winner,MODEL_PATH)
    # Independently reload selected weights and check their FP32 validation predictions.
    batches=loader(dataset,64,4)
    actual={}
    for name in MODEL_NAMES:
        model=load_model(MODEL_PATH,name).to(device='cuda',memory_format=model_memory_format(name))
        _,values=validate(model,batches,torch.device('cuda'))
        assert np.array_equal(np.asarray(values),np.asarray(winner_prob[name]))
        actual[name]=values
        np.save(RUN/f'{name}_champion_probabilities.npy',values)
        del model
    selection=winner['inference_selection']
    chosen=calculate_metrics(dataset.labels,np.mean([actual[n] for n in selection['model_names']],axis=0))
    assert abs(chosen['auroc']-winner_auc)<1e-12
    record.update(selection=selection,model_sha256=sha(MODEL_PATH),post_selection={
        'sources':winner_sources,'improved_over_online_selection':changed,'trials':trials,
        'code_sha256':sha(__file__),'test_data_used':False,'independent_fp32_validation_passed':True})
    (RUN/'POST_SELECTION_COMPLETE.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    (RUN/'inference_selection.json').write_text(json.dumps(selection,indent=2),encoding='utf-8')
    print(json.dumps({'auroc':winner_auc,'sources':winner_sources,'improved':changed,'trials':len(trials)}),flush=True)

if __name__=='__main__':
    run()
