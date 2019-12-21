import glob
import os
import warnings
import numpy as np

import pandas as pd

def parse_bre_gfp(directory):
    filenames = glob.glob(os.path.join(directory, '*.csv'))
    processed_data_records = []
    for fname in filenames:
        try:
            processed_data_records.append(process_file(fname, metadata_fxn=bre_gfp_metadata, data_fxn=compute_ratios))
        except:
            warnings.warn(f"Could not process file {fname}. Skipping...")
            
    df = pd.DataFrame(processed_data_records)
    df = df.reset_index()

    return pd.DataFrame(processed_data_records)

def bre_gfp_metadata(fname_path):
    # Extract file name without preceding path
    fname = os.path.split(fname_path)[-1]
    
    # Drop suffix
    fname = fname[:fname.rfind('.')]
    
    fields = fname.split('_')
    
    metadata = {
        'date': fields[0], 
        'treatment': fields[1],
        'concentration': fields[2],
        'fluorescent channels': fields[3],
        'embryo number': fields[4],
        'somite stage': fields[5],
        'section number': fields[6]
    }
    
    return pd.Series(metadata)

def compute_ratios(df):
    # Mean intensities
    s = df.groupby('Label')['IntDen'].mean()

    # Background subtracted ratios
    ratios = {}
    for label in s.keys():
        ratios[str(label)] = s[label]
    
    # Add metadata about which signal is which
    #ratios['signal'] = signal
    #ratios['normalization signal'] = normalization_signal


    return pd.Series(ratios)

def process_file(fname, metadata_fxn=bre_gfp_metadata, data_fxn=compute_ratios):
    metadata = metadata_fxn(fname)
    ratios = data_fxn(pd.read_csv(fname))
            
    return pd.concat((ratios, metadata))