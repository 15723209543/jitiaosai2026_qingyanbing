"""The reviewed submission stays unchanged while a new continuation run is prepared."""
import argparse
import importlib.util
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]

class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.project=Path(self.tmp.name)
        code=self.project/'submission/code';code.mkdir(parents=True)
        path=code/'train.py';path.write_text((ROOT/'submission/code/train.py').read_text(encoding='utf-8'),encoding='utf-8')
        spec=importlib.util.spec_from_file_location('launcher_under_test',path)
        self.launcher=importlib.util.module_from_spec(spec);spec.loader.exec_module(self.launcher)
        model=self.project/'submission/model/best_model.pth';model.parent.mkdir();model.write_bytes(b'immutable-reviewed-model')
        for variant in ['submission','submission2']:
            c=self.project/'training/20261028/source'/variant/'code';c.mkdir(parents=True)
            (c/'model.py').write_text('# immutable upstream code',encoding='utf-8')
        self.called=[]
        self.runner=types.SimpleNamespace(run=lambda args:self.called.append(args))
        self.addCleanup(lambda:sys.path.remove(str(self.project/'training/20261028')) if str(self.project/'training/20261028') in sys.path else None)

    def invoke(self,*args):
        with patch.dict(sys.modules,{'run_training':self.runner}),patch.object(sys,'argv',['train.py',*args]):self.launcher.main()

    def test_fresh_run_snapshots_source_and_preserves_official_model(self):
        self.invoke('--max-epochs','2')
        run=self.runner.RUN
        self.assertTrue(run.is_relative_to(self.project/'runs'))
        self.assertEqual((run/'source/submission/model/best_model.pth').read_bytes(),b'immutable-reviewed-model')
        self.assertEqual((self.project/'submission/model/best_model.pth').read_bytes(),b'immutable-reviewed-model')
        self.assertEqual(self.called[0].max_epochs,2)
        self.assertFalse(self.called[0].resume)

    def test_resume_uses_existing_run_and_does_not_create_another(self):
        run=self.project/'runs/example';(run/'source/submission/model').mkdir(parents=True)
        (run/'source/submission/model/best_model.pth').write_bytes(b'prior-source')
        (run/'config.json').write_text('{}',encoding='utf-8')
        self.invoke('--resume','runs/example')
        self.assertEqual(self.runner.RUN,run.resolve())
        self.assertTrue(self.called[0].resume)
        self.assertEqual(len(list((self.project/'runs').iterdir())),1)

    def test_resume_outside_runs_is_rejected_before_runner_is_called(self):
        with self.assertRaises(ValueError):self.invoke('--resume','submission')
        self.assertEqual(self.called,[])

if __name__=='__main__':unittest.main()
