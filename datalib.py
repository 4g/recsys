import pandas as pd
from pathlib import Path
import random

class Fields:
  article_id = 'article_id'
  customer_id = 'customer_id'
  time_stamp = 't_dat'
  sales = 'sales'
  start_date = 'start_date'
  end_date = 'end_date'

class Datastore:
  def __init__(self, articles=None, customers=None, transactions=None):
    self.articles = articles
    self.customers = customers
    self.transactions = transactions

  def create_stats(self):
    self.start_date = self.transactions[Fields.time_stamp].min()
    self.end_date = self.transactions[Fields.time_stamp].max()

  def load_from_dir(self, base_path):
    article_path = f'{base_path}/articles.parquet'
    customers_path = f'{base_path}/customers.parquet'
    transactions_path = f'{base_path}/transactions_train.parquet'

    self.transactions = pd.read_parquet(transactions_path, engine="pyarrow")
    self.transactions[Fields.time_stamp] = pd.to_datetime(self.transactions[Fields.time_stamp], infer_datetime_format=True)
    self.customers = pd.read_parquet(customers_path, engine="pyarrow")
    self.articles = pd.read_parquet(article_path, engine="pyarrow")
    self.create_stats()

    print(f"Loaded {base_path}")
    return self

  def save_to_dir(self, base_path):
    Path(base_path).mkdir(exist_ok=True)

    article_path = f'{base_path}/articles.parquet'
    customers_path = f'{base_path}/customers.parquet'
    transactions_path = f'{base_path}/transactions_train.parquet'

    self.transactions.to_parquet(transactions_path, engine="pyarrow")
    self.articles.to_parquet(article_path, engine="pyarrow")
    self.customers.to_parquet(customers_path, engine="pyarrow")

  def sample(self, fraction: float, seed: int):
    small_customers_df = self.customers.sample(frac=fraction, random_state=seed)
    customer_ids = set(small_customers_df[Fields.customer_id])
    small_transaction_df_locs = self.transactions[Fields.customer_id].isin(customer_ids)
    small_transaction_df = self.transactions[small_transaction_df_locs]
    ds = Datastore(articles=self.articles, customers=small_customers_df, transactions=small_transaction_df)
    ds.create_stats()
    return ds

  def get_random_slice(self, num_days, seed=None):
    if seed:
      random.seed(seed)

    total_days = (self.end_date - self.start_date).days
    random_start = random.randint(1, total_days - num_days)
    start_date = self.start_date + pd.Timedelta(days=random_start - 1)
    end_date = start_date + pd.Timedelta(days=num_days + 1)
    _, random_slice = self.split_by_date(start_date)
    random_slice, _ = random_slice.split_by_date(end_date)
    return random_slice

  def head(self, days):
    dsb, _ = self.split_by_date(self.start_date + pd.Timedelta(days=days))
    return dsb

  def tail(self, days):
    _, dsa = self.split_by_date(self.end_date - pd.Timedelta(days=days))
    return dsa

  def split_by_date(self, date):
    before_transactions = self.transactions.loc[self.transactions[Fields.time_stamp] <= date]
    after_transactions = self.transactions.loc[self.transactions[Fields.time_stamp] > date]
    ds_before = Datastore(customers=self.customers, articles=self.articles, transactions=before_transactions)
    ds_after = Datastore(customers=self.customers, articles=self.articles, transactions=after_transactions)
    ds_before.create_stats()
    ds_after.create_stats()
    return ds_before, ds_after

  def info(self):
    self.start_date = self.transactions[Fields.time_stamp].min()
    self.end_date = self.transactions[Fields.time_stamp].max()

    n_days = (self.end_date - self.start_date).days
    d2s = lambda x:f"{x.day}/{x.month}/{x.year}"
    info = {"n_transactions": len(self.transactions),
            "n_customers": len(self.customers),
            "n_articles": len(self.articles),
            "start": d2s(self.start_date),
            "end": d2s(self.end_date),
            "n_days": n_days}

    return info

  def join_all(self):
    new_df = self.transactions.join(self.articles.set_index('article_id'), on='article_id')
    new_df = new_df.join(self.customers.set_index('customer_id'), on='customer_id')
    return new_df

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default=None, help="Load from input dir", required=True)
    parser.add_argument("--sample_dir", default=None, help="Sample the data and save to this dir", required=False)
    parser.add_argument("--seed", default=42, help="Seed to be used for sampling", required=False)
    parser.add_argument("--frac", default=0.1, help="fraction of data to be saved", required=False)

    args = parser.parse_args()
    ds = Datastore().load_from_dir(args.dir)
    print(ds.info())

    if args.sample_dir:
      seed = int(args.seed)
      fraction = float(args.frac)
      ds.sample(fraction=fraction, seed=seed).save_to_dir(args.sample_dir)
