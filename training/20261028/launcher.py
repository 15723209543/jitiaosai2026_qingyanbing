"""Warm-start the reviewed model in an isolated project run; preserve the submission."""
import argparse
from datetime import datetime
import hashlib
from pathlib import Path
import shutil
import sys

def main():
    # This launcher is installed at submission/code/train.py.
    project=Path(__file__).resolve().parents[2]
    template=project/'training/20261028'
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--resume',type=Path,help='existing project runs/<name>')
    p.add_argument('--max-epochs',type=int,default=3)
    args=p.parse_args()
    if args.max_epochs<1:raise ValueError('max-epochs must be positive')
    if args.resume:
        run=(project/args.resume).resolve() if not args.resume.is_absolute() else args.resume.resolve()
        if not run.is_relative_to(project/'runs'):raise ValueError('Resume path must be inside project/runs')
        if not (run/'config.json').is_file():raise FileNotFoundError('Existing run/config.json required')
    else:
        run=project/'runs'/('continuation_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
        run.mkdir(parents=True,exist_ok=False)
        for variant in ['submission','submission2']:
            shutil.copytree(template/'source'/variant/'code',run/'source'/variant/'code',ignore=shutil.ignore_patterns('__pycache__'))
        target=run/'source/submission/model/best_model.pth';target.parent.mkdir(parents=True)
        shutil.copy2(project/'submission/model/best_model.pth',target)
    sys.path.insert(0,str(template))
    import run_training as runner
    runner.RUN=run;runner.PROJECT=project
    runner.SOURCE_SHA=hashlib.sha256((run/'source/submission/model/best_model.pth').read_bytes()).hexdigest()
    print('Project run:',run,flush=True)
    print('Official submission stays unchanged until a completed run is reviewed.',flush=True)
    runner.run(argparse.Namespace(resume=bool(args.resume),max_epochs=args.max_epochs))

if __name__=='__main__':main()
