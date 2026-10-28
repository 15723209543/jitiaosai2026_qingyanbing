"""Local-only YOLO classification supplement; never publish images or label manifests."""
import hashlib,json,math
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from PIL import Image

def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()

def read_extra(root):
    root=Path(root);config=(root/'data.yaml').read_text(encoding='utf-8-sig')
    if "['Glaucoma', 'No Glaucoma']" not in config:raise ValueError('Expected explicit class order Glaucoma, No Glaucoma')
    records=[];splits={}
    for split in ['train','valid','test']:
        images=root/split/'images';labels=root/split/'labels'
        if not images.is_dir() or not labels.is_dir():raise FileNotFoundError(split)
        start=len(records)
        for p in sorted(images.iterdir()):
            if p.suffix.lower() not in ['.jpg','.jpeg','.png']:continue
            rows=[line.split() for line in (labels/(p.stem+'.txt')).read_text(encoding='utf-8-sig').splitlines() if line.strip()]
            classes={r[0] for r in rows}
            if len(classes)!=1 or not classes.issubset({'0','1'}):raise ValueError('Ambiguous/empty label: '+p.name)
            for row in rows:
                if len(row)!=5 or not all(math.isfinite(float(x)) and 0<=float(x)<=1 for x in row[1:]):raise ValueError('Invalid box: '+p.name)
            records.append((p,1-int(next(iter(classes)))))
        splits[split]=len(records)-start
    if not records or {y for p,y in records}!={0,1}:raise ValueError('Both classes required')
    return records,splits

def audit_extra(root,project,run):
    records,splits=read_extra(root)
    valid=json.loads((project/'work/cache384/valid.json').read_text())['sha256']
    test_paths=list((project/'glaucoma_first/test').rglob('*.JPG'))
    if len(test_paths)!=5855:raise ValueError('Expected official test images for exact-duplicate audit')
    with ThreadPoolExecutor(max_workers=8) as pool:protected=set(valid)|set(pool.map(digest,test_paths))
    def check(record):
        path,label=record
        with Image.open(path) as im:im.verify()
        return digest(path)
    with ThreadPoolExecutor(max_workers=8) as pool:hashes=list(pool.map(check,records))
    seen={};kept=[];excluded=0
    for record,h in zip(records,hashes):
        if h in protected:excluded+=1;continue
        if h in seen:
            if seen[h]!=record[1]:raise ValueError('Conflicting labels on duplicate images')
            excluded+=1;continue
        seen[h]=record[1];kept.append(record)
    fingerprint=hashlib.sha256(''.join(h+str(y) for h,y in sorted(seen.items())).encode()).hexdigest()
    summary={'source_splits':splits,'raw_images':len(records),'usable_unique_images':len(kept),
        'positive':sum(y for p,y in kept),'negative':sum(1-y for p,y in kept),'excluded_exact_duplicates':excluded,
        'label_map':{'0_Glaucoma':1,'1_No_Glaucoma':0},'dataset_fingerprint':fingerprint,
        'all_three_extra_splits_used_only_for_training':True,'official_valid_and_test_not_trained':True,
        'patient_level_overlap':'Cannot verify without source patient identifiers',
        'data_publication':'No images, YOLO labels, raw filenames, ZIPs or per-image manifest to GitHub'}
    audit_path=run/'extra_data_audit.json'
    if audit_path.exists():
        prior=json.loads(audit_path.read_text(encoding='utf-8'))
        if prior['dataset_fingerprint']!=fingerprint:raise ValueError('Extra dataset changed since the saved run; start a new run')
    audit_path.write_text(json.dumps(summary,indent=2),encoding='utf-8')
    return kept,summary
