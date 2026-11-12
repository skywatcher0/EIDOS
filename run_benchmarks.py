"""Train and evaluate EIDOS on the bundled regression datasets."""
import os
# Set before importing numerical libraries, including in spawned Windows workers.
for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:
    os.environ[name]='1'
os.environ['PYTHONHASHSEED']='0'

from pathlib import Path
import argparse
import concurrent.futures
import hashlib
import importlib.metadata
import json
import platform
import sys
import time
import numpy as np
import pandas as pd
from benchmarks.common import ROOT, write_json
from benchmarks.jobs import execute

def fingerprint():
    h=hashlib.sha256()
    h.update(json.dumps({p:importlib.metadata.version(p) for p in ['eidos-toolkit','numpy','pandas','scipy','scikit-learn','threadpoolctl']},sort_keys=True).encode())
    h.update((sys.version + platform.platform()).encode())
    h.update(Path(__file__).read_bytes())
    for folder in ['src','benchmarks','configs']:
        for p in sorted((ROOT/folder).rglob('*'),key=lambda p:p.relative_to(ROOT).as_posix()):
            if p.is_file() and p.suffix in {'.py','.json'}:
                h.update(p.relative_to(ROOT).as_posix().encode()); h.update(p.read_bytes())
    for folder in ['data/pmlb','data/openml']:
        for p in sorted((ROOT/folder).rglob('*'),key=lambda p:p.relative_to(ROOT).as_posix()):
            if p.is_file():
                h.update(p.relative_to(ROOT).as_posix().encode()); h.update(p.read_bytes())
    return h.hexdigest()

def build_jobs(suite,out):
    pmlb=pd.read_csv(ROOT/'data/pmlb/catalogue.csv').dataset.tolist()
    real=pd.read_csv(ROOT/'data/openml/catalogue.csv').dataset.tolist()
    groups={}
    if suite in ['all','benchmark','smoke']:
        names=['feynman_I_6_2a','706_sleuth_case1202'] if suite=='smoke' else pmlb
        names=[n for n in names if n in pmlb]
        groups['benchmark']=[('benchmark',(n,0,str(out),'pmlb',True)) for n in names]
    if suite in ['all','openml','smoke']:
        groups['openml']=[('benchmark',(n,p,str(out),'openml',False)) for n in (real[:1] if suite=='smoke' else real) for p in (range(1) if suite=='smoke' else range(10))]
    return groups

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite',choices=['smoke','all','benchmark','openml'],default='all')
    parser.add_argument('--workers',type=int,default=min(4,os.cpu_count() or 1))
    parser.add_argument('--output',type=Path)
    parser.add_argument('--resume',action='store_true',help='Reuse only successful cells with matching source/config fingerprint.')
    args=parser.parse_args()
    if args.workers<1: parser.error('--workers must be positive')
    out=(args.output or ROOT/('results_smoke' if args.suite=='smoke' else 'results')).resolve()
    if out == ROOT or any(out.is_relative_to(ROOT/p) for p in ['src','benchmarks','configs','data','reference','tests','docs','examples','.git']):
        parser.error('Choose a result directory outside the source, data and reference folders')
    out.mkdir(parents=True,exist_ok=True)
    source_hash=fingerprint()
    groups=build_jobs(args.suite,out)
    versions={p:importlib.metadata.version(p) for p in ['numpy','pandas','scipy','scikit-learn','threadpoolctl']}
    write_json(out/'environment.json',{'python':sys.version,'platform':platform.platform(),'packages':versions,
            'workers':args.workers,'blas_threads_per_worker':1,'source_config_sha256':source_hash,'suite':args.suite})
    started=time.perf_counter()
    for name,jobs in groups.items():
        directory=out/name
        directory.mkdir(exist_ok=True)
        cells=directory/'cells'; cells.mkdir(exist_ok=True)
        rows=[]; pending=[]
        for job in jobs:
            # Output location does not affect a scientific cell's identity.
            params=list(job[1]); params[2]=None
            key=hashlib.sha256(json.dumps([job[0],params],sort_keys=True).encode()).hexdigest()
            path=cells/(key+'.json')
            try:
                cached=json.loads(path.read_text(encoding='utf-8')) if args.resume and path.exists() else None
            except (ValueError, OSError):
                cached=None
            artifact=directory/'artifacts'/f'{job[1][0]}__{job[1][1]}'
            artifacts_exist=all((artifact/p).is_file() for p in ['model.json','record.json','prepared.csv','test_predictions.csv','preprocessing.json','payload.bin'])
            if cached and artifacts_exist and cached.get('fingerprint')==source_hash and cached.get('rows') and all(not r.get('failed') for r in cached['rows']):
                rows.extend(cached['rows'])
            else: pending.append((job,path))
        print(f'{name}: {len(pending)} fresh cells; {len(jobs)-len(pending)} cached',flush=True)
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures={pool.submit(execute,job):path for job,path in pending}
            for i,future in enumerate(concurrent.futures.as_completed(futures),1):
                got=future.result()
                write_json(futures[future],{'fingerprint':source_hash,'rows':got})
                rows.extend(got)
                if i%25==0 or i==len(pending) or any(r.get('failed') for r in got):
                    print(f'{name}: {i}/{len(pending)} cells finished; elapsed {time.perf_counter()-started:.1f}s',flush=True)
                    for r in got:
                        if r.get('failed'): print(r.get('error',r),flush=True)
        table=pd.DataFrame(rows)
        keys=[c for c in ['dataset','partition'] if c in table]
        if keys: table=table.sort_values(keys)
        table.to_csv(directory/'raw.csv',index=False)
    from benchmarks.reporting import report
    status=report(out,list(groups),args.suite,source_hash)
    print(json.dumps({'output':str(out),'verification_passed':status,'elapsed_seconds':round(time.perf_counter()-started,2)},ensure_ascii=True),flush=True)
    return 0 if status else 1

if __name__=='__main__':
    raise SystemExit(main())
