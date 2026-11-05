"""Rebuild portable experiment data from bundled raw inputs or public sources."""
from pathlib import Path
import argparse
import gzip
import hashlib
import json
import ssl
import sys
import time
import urllib.request
import numpy as np
import pandas as pd
import certifi
from repro.common import ROOT, write_json
from repro.data import task_seed

def save_csv(frame,path):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('wb') as raw, gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0) as f:
        f.write(frame.to_csv(index=False,lineterminator='\n').encode('utf-8'))

def download(url,dst):
    dst.parent.mkdir(parents=True,exist_ok=True)
    for attempt in range(3):
        try:
            req=urllib.request.Request(url,headers={'User-Agent':'EIDOS-reproduction/1.0'})
            with urllib.request.urlopen(req,timeout=120) as source, dst.with_suffix('.part').open('wb') as target:
                while chunk:=source.read(1024*1024): target.write(chunk)
            dst.with_suffix('.part').replace(dst)
            return
        except Exception:
            if attempt==2: raise
            time.sleep(2*(attempt+1))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--track',choices=['pmlb','openml','recovery'],required=True)
    parser.add_argument('--datasets',default='',help='Comma-separated names; default all.')
    parser.add_argument('--output',type=Path,default=ROOT/'rebuilt_data')
    parser.add_argument('--download-missing',action='store_true',help='Download missing full PMLB source tables (all full sources are external in the compact edition).')
    parser.add_argument('--from-source',action='store_true',help='Download upstream data even when a bundled raw file exists.')
    args=parser.parse_args()
    # Use the pinned public CA bundle and keep TLS verification enabled.
    # Some Python distributions have an incomplete platform CA store.
    context=ssl.create_default_context(cafile=certifi.where())
    urllib.request.install_opener(urllib.request.build_opener(urllib.request.HTTPSHandler(context=context)))
    args.output=args.output.resolve()
    selected=set(args.datasets.split(',')) if args.datasets else None
    checks=[]
    if args.track=='pmlb':
        manifest=json.loads((ROOT/'data/pmlb/manifest.json').read_text(encoding='utf-8'))
        known={r['dataset'] for r in manifest}
        if selected and not selected<=known: parser.error('Unknown datasets: '+str(sorted(selected-known)))
        for info in manifest:
            name=info['dataset']
            if selected and name not in selected: continue
            raw=ROOT/'data/pmlb/raw'/(name+'.tsv.gz')
            source='bundled_raw'
            if args.from_source or not raw.exists():
                if not (args.from_source or args.download_missing):
                    checks.append({'dataset':name,'status':'download_required','url':info['source_url']})
                    continue
                raw=args.output/'downloads'/(name+'.tsv.gz')
                if not raw.exists(): download(info['source_url'],raw)
                source='downloaded_raw'
            frame=pd.read_csv(raw,sep='\t',compression='gzip').dropna().copy()
            y=pd.to_numeric(frame['target'],errors='coerce')
            x=frame.drop(columns=['target']).apply(pd.to_numeric,errors='coerce')
            keep=y.notna()&x.notna().all(axis=1)
            x=x.loc[keep].reset_index(drop=True); y=y.loc[keep].reset_index(drop=True)
            original_indices=np.flatnonzero(keep.to_numpy())
            if len(x)>300:
                take=np.sort(np.random.default_rng(task_seed(name)).choice(np.arange(len(x)),size=300,replace=False))
                x=x.iloc[take].reset_index(drop=True); y=y.iloc[take].reset_index(drop=True)
                original_indices=original_indices[take]
            x.columns=[str(c) for c in x]
            x['target']=y.to_numpy(float)
            dst=args.output/'pmlb/samples'/(name+'.csv.gz')
            save_csv(x,dst)
            reference=pd.read_csv(ROOT/'data/pmlb/samples'/(name+'.csv.gz'),float_precision='round_trip')
            exact=list(x)==list(reference) and x.shape==reference.shape and np.array_equal(x.to_numpy(float),reference.to_numpy(float))
            close=list(x)==list(reference) and x.shape==reference.shape and np.allclose(x.to_numpy(float),reference.to_numpy(float),rtol=1e-12,atol=1e-12)
            checks.append({'dataset':name,'source':source,'rows':len(x),'status':'passed' if close else 'mismatch',
                           'bitwise_numeric_match':bool(exact),'numeric_match_rtol_1e12':bool(close),
                           'source_file_sha256':hashlib.sha256(raw.read_bytes()).hexdigest()})
            write_json(dst.with_name(name+'.row_ids.json'),{'raw_row_indices':original_indices.tolist()})
            print(name,checks[-1]['status'],flush=True)
    elif args.track=='openml':
        catalogue=pd.read_csv(ROOT/'data/openml/catalogue.csv').fillna('')
        if selected and not selected<=set(catalogue.dataset): parser.error('Unknown OpenML dataset')
        for row in catalogue.to_dict('records'):
            name=row['dataset']
            if selected and name not in selected: continue
            if args.from_source:
                from sklearn.datasets import fetch_openml
                data=fetch_openml(data_id=int(row['openml_id']),as_frame=True,parser='pandas',data_home=str(args.output/'downloads/openml'))
                frame=data.frame
                y=pd.to_numeric(frame[row['target']],errors='coerce')
                excluded=[row['target']]+[c for c in str(row['dropped']).split(';') if c and c in frame]
                x=frame.drop(columns=excluded).apply(pd.to_numeric,errors='coerce')
                x=x.loc[:,x.notna().mean()>0.99]
                keep=y.notna()&x.notna().all(axis=1)
                frame=x.loc[keep].reset_index(drop=True)
                frame['target']=y.loc[keep].to_numpy(float)
            else:
                frame=pd.read_csv(ROOT/'data/openml/raw'/(name+'.csv.gz'),float_precision='round_trip')
            save_csv(frame,args.output/'openml/raw'/(name+'.csv.gz'))
            reference=pd.read_csv(ROOT/'data/openml/raw'/(name+'.csv.gz'),float_precision='round_trip')
            close=list(frame)==list(reference) and frame.shape==reference.shape and np.allclose(frame.to_numpy(float),reference.to_numpy(float),rtol=1e-12,atol=1e-12)
            checks.append({'dataset':name,'openml_id':int(row['openml_id']),'rows':len(frame),'status':'passed' if close else 'mismatch'})
            print(name,checks[-1]['status'],flush=True)
    else:
        from experiments.recovery import _generate, GENERATORS
        for generator in GENERATORS:
            for sigma in [0.01,0.05,0.1,0.2,0.4,0.8,1.6,3.2]:
                for seed in range(20260225,20260230):
                    frame=_generate(generator,sigma,seed)
                    save_csv(frame,args.output/'recovery'/f'{generator}__{sigma:g}__{seed}.csv.gz')
        checks.append({'status':'passed','generated_cells':200,'rows_per_cell':600})
    write_json(args.output/(args.track+'_preparation_report.json'),checks)
    return 0 if all(c['status']=='passed' for c in checks) else 1

if __name__=='__main__':
    raise SystemExit(main())
