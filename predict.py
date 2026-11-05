"""Predict from an emitted model record, without fitting or the original data."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from repro.records import unpack

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--input',type=Path,required=True,help='CSV containing original named features, or model feature columns with --prepared.')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--prepared',action='store_true',help='Input already uses the model feature names and discovery-standardised units.')
    args=parser.parse_args()
    record=json.loads(args.model.read_text(encoding='utf-8'))
    model=unpack(record)
    table=pd.read_csv(args.input,float_precision='round_trip')
    context=record.get('construction_context') or {}
    if args.prepared or 'feature_affine' not in context:
        x=table[model.feature_cols].to_numpy(float)
    else:
        x=table[context['selected_original_features']].to_numpy(float)
        for i,col in enumerate(model.feature_cols):
            center,scale=context['feature_affine'][col]
            x[:,i]=(x[:,i]-center)/scale
    prediction,sigma=model.predict_mu_sigma(x)
    result=pd.DataFrame({'prediction_model_units':prediction,'sigma_model_units':sigma})
    if 'target_affine' in context:
        center,scale=context['target_affine']
        result['prediction_raw_units']=prediction*scale+center
    args.output.parent.mkdir(parents=True,exist_ok=True)
    result.to_csv(args.output,index=False)
    print(f'Predicted {len(result)} rows from the saved model.')
    return 0

if __name__=='__main__':
    raise SystemExit(main())
