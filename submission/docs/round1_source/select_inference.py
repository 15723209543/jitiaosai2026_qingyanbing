"""Lock the best validation candidate before any final test inference."""
import argparse
import json
from pathlib import Path
import torch
from utils import MODEL_PATH, PROJECT_DIR, RECIPE

def select(run_dir):
    metrics = json.loads((run_dir / 'validation_metrics.json').read_text(encoding='utf-8'))
    completed = json.loads((run_dir / 'TRAINING_COMPLETE.json').read_text(encoding='utf-8'))
    checkpoint = torch.load(MODEL_PATH, map_location='cpu', weights_only=True)
    if checkpoint.get('training_complete') is not True or completed['epochs_each'] != checkpoint['epochs_trained_each']:
        raise RuntimeError('Both branches must finish training before inference selection')
    for name in ('resnet18', 'densenet121'):
        branch = checkpoint['models'][name]
        if branch['model_config']['recipe'] != RECIPE or abs(branch['validation']['auroc']-metrics[name]['auroc']) > 1e-6:
            raise ValueError('Validation record does not match checkpoint')
    candidates = [(['resnet18', 'densenet121'], 'fixed_mean_ensemble'),
                  (['resnet18'], 'resnet18'), (['densenet121'], 'densenet121')]
    names, key = max(candidates, key=lambda pair: metrics[pair[1]]['auroc'])
    selection = {'model_names': names, 'metric': 'official_validation_auroc',
                 'candidate_metrics': {key: metrics[key] for _, key in candidates},
                 'selected_candidate': key, 'validation': metrics[key],
                 'rule': 'best of two individual models and fixed equal mean; no test data read',
                 'threshold': 0.5, 'tta': False, 'inference_precision': 'FP32'}
    checkpoint['inference_selection'] = selection
    temporary = MODEL_PATH.with_suffix('.tmp')
    torch.save(checkpoint, temporary)
    temporary.replace(MODEL_PATH)
    (run_dir / 'inference_selection.json').write_text(json.dumps(selection, indent=2), encoding='utf-8')
    print(json.dumps(selection, indent=2), flush=True)
    return selection

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, default=PROJECT_DIR/'runs'/'20261024_upstream_dual')
    select(parser.parse_args().run_dir)
