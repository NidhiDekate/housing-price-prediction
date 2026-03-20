import os
import pandas as pd

BASE      = '/content/drive/MyDrive/Housing_Project'
RAW       = os.path.join(BASE, 'data/raw')
PROCESSED = os.path.join(BASE, 'data/processed')
MODELS    = os.path.join(BASE, 'data/models')

def load_raw():
    ames = pd.read_csv(os.path.join(RAW, 'AmesHousing.csv'))
    kc   = pd.read_csv(os.path.join(RAW, 'kc_house_data.csv'))
    return ames, kc

def load_processed(filename):
    return pd.read_csv(os.path.join(PROCESSED, filename))

def save_processed(df, filename):
    df.to_csv(os.path.join(PROCESSED, filename), index=False)
    print(f"Saved: {filename}")
