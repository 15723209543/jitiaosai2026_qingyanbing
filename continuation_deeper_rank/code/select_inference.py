"""Verify the already locked first-improvement snapshot, without further model search."""
import argparse,hashlib,json
from pathlib import Path
import torch
from utils import PROJECT_DIR,MODEL_PATH,assert_inference_checkpoint_ready
def select(run_dir):
    record=json.loads((run_dir/'POST_SELECTION_COMPLETE.json').read_text())
    assert hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()==record['model_sha256']
    checkpoint=torch.load(MODEL_PATH,map_location='cpu',weights_only=True)
    names=checkpoint['inference_selection']['model_names']
    assert_inference_checkpoint_ready(MODEL_PATH,names)
    if 'resnet34_graft' in names:assert checkpoint['models']['resnet34_graft']['student_trained_epochs']>=1
    print(json.dumps(record['selection'],indent=2),flush=True)
    return record['selection']
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,default=PROJECT_DIR/'runs/20261025_deeper_rank');select(p.parse_args().run_dir)
