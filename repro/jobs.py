import time
from pathlib import Path
import numpy as np
import pandas as pd
from .common import ROOT, NullLedger, config, digest, canonical_bytes, write_json
from .data import prepare, task_seed
from .records import pack, unpack
from src.eidos.search.engine import run_eidos_dataset

def fit(frame, meta, name, cfg=None):
    cfg = config('benchmark') if cfg is None else cfg
    cfg['search']['feature_affine'] = meta['feature_affine']
    cfg['search']['target_affine'] = meta['target_affine']
    return run_eidos_dataset(df=frame,dataset_id=name,feature_cols=[c for c in frame if c.startswith('x')],
        target_col='y_tilde',env_col='env_id',split_col='split',delta_y=0.001,
        seed=meta['seed'],cfg=cfg,ledger=NullLedger(),run_id='reproduction',git_hash='archived',audit_trace=True)

def model_record(result, meta):
    # The context contains only discovery-derived transformations and fixed seeds.
    context = {k:meta[k] for k in ['seed','feature_affine','target_affine','selected_original_features']}
    return pack(result.best_model,float(result.layer_stats['final_param_step']),result.selection_trace,context)

def benchmark(name, partition, out, track='pmlb', conformance=False):
    started = time.perf_counter()
    frame, meta = prepare(name,track,partition)
    if track == 'openml':
        meta['seed'] %= 2**31-1
    result = fit(frame,meta,name)
    record = model_record(result,meta)
    test = frame[frame['split']=='test']
    decoded = unpack(record)
    preds, _ = decoded.predict_mu_sigma(test[decoded.feature_cols].to_numpy(float))
    if not np.array_equal(preds,result.test_pred):
        raise AssertionError('Model round-trip changed predictions')
    directory = Path(out)/('benchmark' if track=='pmlb' else 'openml')/'artifacts'/f'{name}__{partition}'
    directory.mkdir(parents=True,exist_ok=True)
    write_json(directory/'model.json',record)
    write_json(directory/'preprocessing.json',meta)
    (directory/'payload.bin').write_bytes(canonical_bytes(record))
    frame.to_csv(directory/'prepared.csv',index=False)
    pd.DataFrame({'sample_id':result.test_sample_ids,'target':result.test_y,'prediction':result.test_pred,
        'lower':result.test_lower,'upper':result.test_upper,
        'prediction_raw_units':result.test_pred*meta['target_affine'][1]+meta['target_affine'][0]}).to_csv(directory/'test_predictions.csv',index=False)
    stats=result.layer_stats
    row={'dataset':name,'partition':partition,'rmse':result.rmse_test,
         'rmse_raw_units':result.rmse_test*meta['target_affine'][1],
         'rmse_full_precision':stats['rmse_test_full_precision'],'candidate_id':result.best_candidate_id,
         'p_inv_lock':result.p_inv_lock,'coverage':result.conformal_coverage,'interval_length':result.conformal_length,
         'final_n_basis':stats['final_n_basis'],'final_param_step':stats['final_param_step'],
         'expr':result.best_model.to_expr().to_string(), 'payload_sha256':digest(record),
         'decode_exact':True, 'failed':False,'runtime_sec':time.perf_counter()-started}
    if conformance:
        replaced=frame.copy()
        later=replaced['split'].isin(['audit','cal','test'])
        rng=np.random.default_rng(task_seed(name)+1)
        replaced.loc[later,'y_tilde']=rng.permutation(replaced.loc[later,'y_tilde'].to_numpy())+rng.normal(0,1,int(later.sum()))
        second=fit(replaced,meta,name)
        changed=model_record(second,meta)
        row['replacement_identical']=canonical_bytes(record)==canonical_bytes(changed)
        row['replacement_sha256']=digest(changed)
        write_json(directory/'replacement_check.json',{'identical':row['replacement_identical'],
                   'original_sha256':row['payload_sha256'],'replacement_sha256':row['replacement_sha256'],
                   'strategy':'joint later-role response permutation plus Gaussian noise; seed=task_seed+1'})
        if not row['replacement_identical']:
            raise AssertionError('Later-role replacement changed construction payload')
    return [row]

def recovery(generator, sigma, seed, out):
    from experiments import recovery as e
    started=time.perf_counter()
    bank, truth_ids=e._build_bank(240)
    frame=e._generate(generator,sigma,seed)
    trace={}
    entry, model, diagnostics=e._select_and_refit(frame,bank,seed,'cv_rmse_then_dl',3,1,trace=trace)
    test=frame[frame['split']=='test']
    pred,_=model.predict_mu_sigma(test[['x1','x2','x3']].to_numpy(float))
    true=test['truth'].to_numpy(float)
    selected_vars,selected_ops=e._symbols(entry.spec.basis_exprs)
    truth=next(x for x in bank if x.spec.candidate_id==truth_ids[generator])
    true_vars,true_ops=e._symbols(truth.spec.basis_exprs)
    record=pack(model,0.001,trace,{'seed':seed,'bank_name':e.BANK_NAME})
    if not np.array_equal(pred,unpack(record).predict_mu_sigma(test[['x1','x2','x3']].to_numpy(float))[0]):
        raise AssertionError('Recovery decoder mismatch')
    directory=Path(out)/'recovery'/'artifacts'/f'{generator}__{sigma:g}__{seed}'
    directory.mkdir(parents=True,exist_ok=True)
    write_json(directory/'model.json',record)
    frame.assign(prediction=np.nan).to_csv(directory/'data.csv',index=False)
    pd.DataFrame({'row':test.index,'truth':true,'target':test.y_tilde,'prediction':pred}).to_csv(directory/'test_predictions.csv',index=False)
    return [{'generator':generator,'noise_sigma':sigma,'seed':seed,'bank_size':len(bank),
        'selected_candidate_id':entry.spec.candidate_id,'truth_candidate_id':truth_ids[generator],
        'recovered':entry.spec.candidate_id==truth_ids[generator],
        'semantic_nrmse':float(np.sqrt(np.mean((pred-true)**2))/max(float(np.std(true)),1e-12)),
        'variable_precision':len(selected_vars&true_vars)/max(len(selected_vars),1),
        'variable_recall':len(selected_vars&true_vars)/max(len(true_vars),1),
        'operator_precision':len(selected_ops&true_ops)/max(len(selected_ops),1),
        'operator_recall':len(selected_ops&true_ops)/max(len(true_ops),1),
        'rmse_test':float(np.sqrt(np.mean((pred-test.y_tilde.to_numpy(float))**2))),
        'expr':model.to_expr().to_string(),'payload_sha256':digest(record),
        'decode_exact':True,'failed':False,'runtime_sec':time.perf_counter()-started,**diagnostics}]

def conformal(generator, arm, seed, out):
    from experiments import conformal as e
    rows=e._run_cell(generator,arm,seed,0.1,e._strict_config(39))
    for row in rows: row['failed']=False
    return rows

def execute(job):
    """Top-level function for Windows spawn as well as Unix multiprocessing."""
    suite, args = job
    try:
        return globals()[suite](*args)
    except Exception:
        import traceback
        return [{'failed':True,'suite':suite,'job':str(args[:-1]),'error':traceback.format_exc()}]
