"""Validate the supplied YOLO class mapping without using or publishing real data."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

spec=importlib.util.spec_from_file_location('extra_dataset',Path(__file__).resolve().parents[1]/'training/20261028/extra_data.py')
extra=importlib.util.module_from_spec(spec);spec.loader.exec_module(extra)

class ExtraDataTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        (self.root/'data.yaml').write_text("names: ['Glaucoma', 'No Glaucoma']",encoding='utf-8')
        for split,label in [('train','0'),('valid','1'),('test','0')]:
            (self.root/split/'images').mkdir(parents=True);(self.root/split/'labels').mkdir()
            (self.root/split/'images/a.jpg').write_bytes(b'placeholder')
            (self.root/split/'labels/a.txt').write_text(label+' 0.5 0.5 0.8 0.8',encoding='utf-8')
    def test_glaucoma_zero_maps_to_positive_one_across_all_training_splits(self):
        records,splits=extra.read_extra(self.root)
        self.assertEqual([y for p,y in records],[1,0,1])
        self.assertEqual(splits,{'train':1,'valid':1,'test':1})
    def test_mixed_or_empty_annotation_is_rejected(self):
        p=self.root/'train/labels/a.txt'
        for value in ['', '0 0.5 0.5 1 1\n1 0.5 0.5 1 1']:
            p.write_text(value,encoding='utf-8')
            with self.assertRaises(ValueError):extra.read_extra(self.root)
    def test_unknown_class_order_is_rejected(self):
        (self.root/'data.yaml').write_text("names: ['No Glaucoma', 'Glaucoma']",encoding='utf-8')
        with self.assertRaises(ValueError):extra.read_extra(self.root)
