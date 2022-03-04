import datetime

from datalib import Fields
from pandas import DataFrame
import pandas as pd
import numpy as np
from tqdm import tqdm

## TODO
"""
1. Generate before after df and check accuracy
2. Simple test : 
    a. Popular unbought products
3. Signals :
    a. Product similarity : Hand crafted
    b. User similarity : hand crafted
    c. User attribute affinity
    d. User trend affinity
"""



def article_popularity(datastore, age_weighted=True, n_days=21):
  datastore.info()
  split_date = datastore.end_date - pd.Timedelta(days=n_days)
  _, datastore_after = datastore.split_by_date(split_date)

  ts = datastore_after.transactions
  decay_mod_days = 3
  decay_rate = 21
  decay_df = 1

  if age_weighted:
    yy = datastore.end_date - ts[Fields.time_stamp]
    decay_df = (1/((yy.dt.days//decay_mod_days) * decay_rate + 1))

  ts[Fields.sales] = decay_df

  article_groups = ts[[Fields.sales, Fields.article_id]].groupby([Fields.article_id])
  article_sales = article_groups.sum()
  return article_sales.reset_index()


def get_bought(datastore):
  customer_groups = datastore.transactions[[Fields.customer_id, Fields.article_id]].groupby([Fields.customer_id])
  bought = customer_groups[Fields.article_id].apply(list).to_dict()
  return bought

def get_repeat_counts(ts, field):
  ts[Fields.sales] = 1
  article_groups = ts[[Fields.sales, field]].groupby([field])
  article_sales = article_groups.sum()
  s = article_sales.reset_index()[[field, Fields.sales]].to_dict(orient='records')
  return {r[field]: r[Fields.sales] for r in s}


def create_submission(recommender, out):
  recommendations = recommender.predict_transactions(top_k=12)
  with open(out, 'w') as outfile:
    header = "customer_id,prediction\n"
    outfile.write(header)

    for customer_id in recommendations:
      rec = ["0" + str(i) for i in recommendations[customer_id]]
      rec = " ".join(rec)
      line = f"{customer_id},{rec}\n"
      outfile.write(line)

    outfile.close()

if __name__ == "__main__":
  import argparse

  parser = argparse.ArgumentParser()
  parser.add_argument("--dir", default=None, help="Load datastore input dir", required=True)
  parser.add_argument("--out", default="/tmp/recommendations.csv", help="Output recommendations file", required=False)

  args = parser.parse_args()

  from datalib import Datastore

  ds = Datastore().load_from_dir(args.dir)
  get_bought(datastore=ds)
