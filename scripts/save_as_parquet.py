from tqdm import tqdm
import pandas as pd
import numpy as np
from time import time

csv_list = [
  'transactions_train',
  'articles',
  'customers',
  'sample_submission'
]

full = "full/"

def reformat():
  for f_name in csv_list:
    f_csv = f"../data/{f_name}.csv"

    print(f"Loading {f_csv}")
    df = pd.read_csv(f_csv)
    f_pqt = f"../{full}/{f_name}.parquet"

    print(f"Saving {f_pqt}")
    df.to_parquet(f_pqt, engine="pyarrow")