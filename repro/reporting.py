"""Independent, post-fit reference comparisons and figures from fresh results."""
from pathlib import Path
import numpy as np
import pandas as pd
from .common import ROOT, write_json

SPECS={
 'benchmark':('benchmark.csv',['dataset'],['rmse','coverage','interval_length','p_inv_lock'],['candidate_id']),
 'openml':('openml.csv',['dataset','partition'],['rmse'],[]),
 'recovery':('recovery_noise_sweep.csv',['generator','noise_sigma','seed'],['semantic_nrmse','rmse_test'],['selected_candidate_id','recovered']),
 'conformal':('conformal_two_arm.csv',['generator','arm','seed','method'],['observed_coverage','mean_interval_length'],['selected_candidate_id']),
}

def report(out,suites,mode,fingerprint):
    checks=[]; frames={}
    for suite in suites:
        actual=pd.read_csv(out/suite/'raw.csv')
        frames[suite]=actual
        errors=int(actual['failed'].fillna(True).astype(bool).sum())
        checks.append({'suite':suite,'check':'successful_rows','passed':errors==0,'failures':errors,'rows':len(actual)})
        file,keys,metrics,categories=SPECS[suite]
        reference=pd.read_csv(ROOT/'reference'/file)
        good=actual[~actual.failed.astype(bool)]
        if not set(keys).issubset(good):
            continue
        paired=good.merge(reference,on=keys,suffixes=('_actual','_reference'),how='left',validate='one_to_one',indicator=True)
        checks.append({'suite':suite,'check':'reference_keys','passed':bool((paired['_merge']=='both').all()) and len(paired)>0})
        if mode!='smoke':
            checks.append({'suite':suite,'check':'full_grid','passed':len(actual)==len(reference),'expected':len(reference),'actual':len(actual)})
        diffs=[]
        for field in metrics:
            a=paired[field+'_actual'].to_numpy(float); b=paired[field+'_reference'].to_numpy(float)
            strict_ok=np.isclose(a,b,rtol=1e-5,atol=1e-8,equal_nan=False)
            # PMLB responses are quantised at delta=0.001. Permit at most 1%
            # of one response bin for cross-platform BLAS variation in distance
            # metrics, while retaining the stricter comparison for inspection.
            atol=1e-5 if suite=='benchmark' and field in ['rmse','interval_length'] else 1e-8
            ok=np.isclose(a,b,rtol=1e-5,atol=atol,equal_nan=False)
            checks.append({'suite':suite,'check':field,'passed':bool(ok.all()),'matched':int(ok.sum()),'total':len(ok),
                          'strict_passed':bool(strict_ok.all()),'strict_matched':int(strict_ok.sum()),
                          'rtol':1e-5,'atol':atol,'strict_atol':1e-8,'max_abs_difference':float(np.max(abs(a-b))) if len(a) else None})
            for i in np.flatnonzero(~strict_ok):
                diffs.append({**{k:paired.iloc[i][k] for k in keys},'field':field,'actual':a[i],'reference':b[i]})
        for field in categories:
            ok=paired[field+'_actual'].astype(str)==paired[field+'_reference'].astype(str)
            checks.append({'suite':suite,'check':field,'passed':bool(ok.all()),'matched':int(ok.sum()),'total':len(ok)})
            for i in np.flatnonzero(~ok.to_numpy()):
                diffs.append({**{k:paired.iloc[i][k] for k in keys},'field':field,'actual':str(paired.iloc[i][field+'_actual']),'reference':str(paired.iloc[i][field+'_reference'])})
        pd.DataFrame(diffs,columns=keys+['field','actual','reference']).to_csv(out/suite/'reference_differences.csv',index=False)
        if 'decode_exact' in good:
            checks.append({'suite':suite,'check':'record_prediction_roundtrip','passed':bool(good.decode_exact.all())})
        if 'replacement_identical' in good:
            checks.append({'suite':suite,'check':'replacement_invariance','passed':bool(good.replacement_identical.all()),
                           'passed_tasks':int(good.replacement_identical.sum()),'tasks':len(good)})
    passed=all(c['passed'] for c in checks)
    write_json(out/'verification_report.json',{'passed':passed,'scope':mode,'source_config_sha256':fingerprint,
            'note':'Timing is excluded. Smoke checks only selected cells. PMLB RMSE/width allow 1% of the 0.001 response bin (atol=1e-5); strict atol=1e-8 matches and differences remain visible. Other metrics retain atol=1e-8.',
            'checks':checks})
    summaries={}
    if 'benchmark' in frames:
        tab=frames['benchmark'].merge(pd.read_csv(ROOT/'data/pmlb/catalogue.csv')[['dataset','task_family','benchmark_role']],on='dataset',how='left')
        summary=tab.groupby('benchmark_role').agg(tasks=('dataset','size'),median_rmse=('rmse','median'),median_runtime_sec=('runtime_sec','median'),median_basis_terms=('final_n_basis','median'),replacement_passes=('replacement_identical','sum'))
        summary.to_csv(out/'benchmark'/'summary.csv'); summaries['benchmark']=summary.reset_index().to_dict('records')
    if 'openml' in frames:
        summary=frames['openml'].groupby('dataset').agg(runs=('rmse','size'),mean_rmse=('rmse','mean'),median_rmse=('rmse','median'))
        summary.to_csv(out/'openml'/'summary.csv'); summaries['openml']=summary.reset_index().to_dict('records')
    if 'recovery' in frames:
        tab=frames['recovery']
        summary=tab.groupby(['generator','noise_sigma']).agg(runs=('recovered','size'),recovery_rate=('recovered','mean'),semantic_nrmse=('semantic_nrmse','mean'),operator_precision=('operator_precision','mean'),operator_recall=('operator_recall','mean'))
        summary.to_csv(out/'recovery'/'summary.csv'); summaries['recovery']=summary.reset_index().to_dict('records')
        tab[(tab.noise_sigma==0.8)&(tab.seed==20260225)&tab.generator.isin(['oscillator','lorentz','blackbody'])].to_csv(out/'recovery'/'formula_examples.csv',index=False)
    if 'conformal' in frames:
        summary=frames['conformal'].groupby(['arm','method']).agg(runs=('seed','size'),coverage=('observed_coverage','mean'),mean_width=('mean_interval_length','mean'))
        summary.to_csv(out/'conformal'/'summary.csv'); summaries['conformal']=summary.reset_index().to_dict('records')
    write_json(out/'summary.json',summaries)
    plots(out,frames)
    return passed

def plots(out,frames):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    if 'recovery' in frames:
        tab=frames['recovery']; fig,axes=plt.subplots(1,2,figsize=(10,3.6),layout='constrained')
        for name,group in tab.groupby('generator'):
            mean=group.groupby('noise_sigma').recovered.mean()
            axes[0].plot(mean.index,mean.values,'o-',label=name)
        axes[0].set(xscale='log',xlabel='Noise standard deviation',ylabel='Exact structure recovery',ylim=(-0.03,1.03))
        axes[0].legend(fontsize=8)
        mean=tab.groupby('noise_sigma')[['operator_precision','operator_recall']].mean()
        axes[1].plot(mean.index,mean.operator_precision,'o-',label='Operator precision')
        axes[1].plot(mean.index,mean.operator_recall,'s-',label='Operator recall')
        axes[1].set(xscale='log',xlabel='Noise standard deviation',ylabel='Mean precision / recall',ylim=(0,1.03)); axes[1].legend()
        fig.savefig(out/'recovery'/'recovery.png',dpi=180); fig.savefig(out/'recovery'/'recovery.pdf'); plt.close(fig)
    if 'conformal' in frames:
        tab=frames['conformal']; groups=tab.groupby(['arm','method']).observed_coverage
        means=groups.mean(); sem=groups.sem().fillna(0)
        fig,ax=plt.subplots(figsize=(8,3.8),layout='constrained')
        ax.bar(range(len(means)),means.values,yerr=1.96*sem.values,color='#287a91',capsize=4)
        ax.axhline(0.9,color='#b54c35',linestyle='--',label='Nominal 0.900')
        ax.set_xticks(range(len(means)),[f'{a}\n{b}' for a,b in means.index],fontsize=8)
        ax.set(ylabel='Mean marginal coverage',ylim=(0,1.05)); ax.legend()
        fig.savefig(out/'conformal'/'coverage.png',dpi=180); fig.savefig(out/'conformal'/'coverage.pdf'); plt.close(fig)
