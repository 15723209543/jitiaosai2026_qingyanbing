"""Verify the validation-selected snapshot already locked by continuation training."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from utils import PROJECT_DIR, MODEL_PATH, assert_inference_checkpoint_ready
from model import MODEL_NAMES

def select(run_dir):
    assert_inference_checkpoint_ready(MODEL_PATH,MODEL_NAMES)
    record=json.loads((run_dir/'TRAINING_COMPLETE.json').read_text(encoding='utf-8'))
    digest=hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()
    if digest!=record['model_sha256']:
        raise ValueError('Completed-run model hash does not match the inference checkpoint')
    checkpoint=torch.load(MODEL_PATH,map_location='cpu',weights_only=True)
    selection=checkpoint['inference_selection']
    print(json.dumps(selection,indent=2),flush=True)
    return selection

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-dir',type=Path,default=PROJECT_DIR/'runs'/'20261024_continue_lr3e5')
    select(parser.parse_args().run_dir)
