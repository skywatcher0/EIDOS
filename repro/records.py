"""Self-contained model records with exact binary64 hex values and a safe decoder.

This JSON transport is not a new MDL code: codelengths remain those computed by
src/eidos/mdl. It retains all state needed to reproduce the scored predictor.
"""
import numpy as np
from src.eidos.dsl.grammar import Expr
from src.eidos.search.models import CandidateSpec, LinearBasisModel
from .common import json_safe

def expr_pack(expr):
    return {'op':expr.op, 'name':expr.name, 'value':float(expr.value).hex() if expr.op=='const' else expr.value,
            'args':[expr_pack(x) for x in expr.args]}

def expr_unpack(obj):
    return Expr(op=obj['op'], name=obj['name'], value=float.fromhex(obj['value']) if obj['op']=='const' else obj['value'],
                args=tuple(expr_unpack(x) for x in obj['args']))

def pack(model, step, trace=None, context=None):
    spec = model.spec
    p = model.param_vector()
    return {'schema':'eidos-reproduction-record-v1', 'feature_cols':model.feature_cols,
            'spec':{'candidate_id':spec.candidate_id,'sigma_mode':spec.sigma_mode,'noise':spec.noise,
                    'df':float(spec.df).hex(),'ridge':float(spec.ridge).hex(),
                    'parent_candidate_id':spec.parent_candidate_id,'basis':[expr_pack(e) for e in spec.basis_exprs]},
            'ridge':float(model.ridge).hex(), 'parameters':[float(v).hex() for v in p],
            'grid_step':float(step).hex(), 'grid_indices':[int(v) for v in np.rint(p/step)],
            'basis_mean':[[float(v).hex() for v in row] for row in model.phi_mean],
            'basis_std':[[float(v).hex() for v in row] for row in model.phi_std],
            'selection_trace':json_safe(trace), 'construction_context':json_safe(context)}

def unpack(record):
    if record.get('schema') != 'eidos-reproduction-record-v1':
        raise ValueError('Unknown model record schema')
    s = record['spec']
    spec = CandidateSpec(candidate_id=s['candidate_id'],basis_exprs=[expr_unpack(e) for e in s['basis']],
                         sigma_mode=s['sigma_mode'],noise=s['noise'],df=float.fromhex(s['df']),
                         ridge=float.fromhex(s['ridge']),parent_candidate_id=s['parent_candidate_id'])
    model = LinearBasisModel(spec,record['feature_cols'],ridge=float.fromhex(record['ridge']),device='cpu')
    model.set_param_vector(np.array([float.fromhex(v) for v in record['parameters']]))
    model.phi_mean = np.array([[float.fromhex(v) for v in row] for row in record['basis_mean']])
    model.phi_std = np.array([[float.fromhex(v) for v in row] for row in record['basis_std']])
    return model
