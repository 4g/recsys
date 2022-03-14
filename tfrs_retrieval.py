import json

import pandas as pd
import  tensorflow as tf
import tensorflow_recommenders as tfrs
from embedders import FieldsModel
import numpy as np
from datalib import Datastore, Fields
from tqdm import tqdm


class RecommenderModel(tfrs.models.Model):
  def __init__(self, article_model, customer_model, articles):
    super().__init__()
    self.candidate_model = tf.keras.Sequential([
      article_model,
      tf.keras.layers.BatchNormalization(),
      tf.keras.layers.Dense(512),
      tf.keras.layers.LeakyReLU(0.01),

      tf.keras.layers.BatchNormalization(),
      tf.keras.layers.Dense(128, activation='relu'),
      tf.keras.layers.LeakyReLU(0.01),
    ])

    self.query_model = tf.keras.Sequential([
      customer_model,
      tf.keras.layers.BatchNormalization(),
      tf.keras.layers.Dense(512),
      tf.keras.layers.LeakyReLU(0.01),

      tf.keras.layers.BatchNormalization(),
      tf.keras.layers.Dense(128, activation='relu'),
      tf.keras.layers.LeakyReLU(0.01),
    ])

    self.task = tfrs.tasks.Retrieval(
        metrics=tfrs.metrics.FactorizedTopK(
            candidates=articles.batch(16384).map(self.candidate_model),
            metrics=[
            tf.keras.metrics.TopKCategoricalAccuracy(
              k=x, name=f"t{x}") for x in [5, 12, 24, 96]]
        ),
    )

  def compute_loss(self, features, training=False):
    query_embeddings = self.query_model(features)
    movie_embeddings = self.candidate_model(features)

    return self.task(query_embeddings, movie_embeddings)

def datastore_to_dataset(datastore, batch_size):
  train_ds = tf.data.Dataset.from_tensor_slices(dict(datastore.transactions))
  train_ds.shuffle(buffer_size=5_000_000, seed=42, reshuffle_each_iteration=True)
  cached_train = train_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
  return cached_train

def train(datastore, train_days, test_days, epochs, article_columns, customer_columns, pretrained=False):
  datastore = datastore.tail(days=train_days + test_days)

  datastore.transactions = datastore.join_all()

  train_datastore, val_datastore = datastore.split_by_date(datastore.end_date - pd.Timedelta(days=test_days))

  print(train_datastore.info())
  print(val_datastore.info())

  all_columns = list(set(article_columns + customer_columns))
  train_datastore.transactions =  train_datastore.transactions[all_columns]
  val_datastore.transactions = val_datastore.transactions[all_columns]

  article_model = FieldsModel(train_datastore.transactions, article_columns, max_tokens=3_000_000)
  customer_model = FieldsModel(train_datastore.transactions, customer_columns, max_tokens=3_000_000)

  articles_df = train_datastore.transactions[article_columns]
  articles_df = articles_df.drop_duplicates(keep='first', subset=[Fields.article_id])
  articles_as_tfds = dict(articles_df)
  articles_as_tfds = tf.data.Dataset.from_tensor_slices(articles_as_tfds)
  recommender_model = RecommenderModel(article_model,
                                       customer_model,
                                       articles_as_tfds)

  train_ds = datastore_to_dataset(train_datastore, batch_size=2048)
  val_ds = datastore_to_dataset(val_datastore, batch_size=8192)

  recommender_model.compile(optimizer=tf.keras.optimizers.Adam(clipnorm=100))
  try:
    recommender_model.fit(train_ds, validation_data=val_ds, epochs=epochs)
  except:
    print("")
    print("... xxxxx ...... ==================== ...... xxxx ...")
    print("... xxxxx ...... Training interrupted ...... xxxx ...")
    print("... xxxxx ...... ==================== ...... xxxx ...")
    pass

  return recommender_model

def model_to_index(query_model, candidate_model, dataset_name, article_columns, test_days, trending_days, top_k, percent_trending):
  from signals import get_repeat_counts
  last_15_days = Datastore().load_from_dir(dataset_name)

  augment_df_timestamp(last_15_days)
  augment_df_product_sales(last_15_days)
  last_15_days = last_15_days.tail(days=trending_days + test_days + 1).head(days=trending_days)
  print("trending datastore", last_15_days.info())
  product_counts = get_repeat_counts(last_15_days.transactions, Fields.article_id)

  sorted_products = sorted(product_counts, key=product_counts.get, reverse=True)
  top_products = sorted_products[:int(len(sorted_products)*percent_trending)]
  top_products_set = set(top_products)
  print("Num top products", len(top_products_set))
  last_15_days.transactions = last_15_days.join_all()
  popular_articles = last_15_days.transactions[last_15_days.transactions.article_id.isin(top_products_set)]
  popular_articles_df = popular_articles[article_columns].drop_duplicates(keep='last', subset=[Fields.article_id])
  top_product_ids = list(popular_articles_df[Fields.article_id].values)
  top_products_tfds = tf.data.Dataset.from_tensor_slices(top_product_ids)
  # print("identifiers", top_product_ids)
  popular_articles_tfds = dict(popular_articles_df)
  popular_articles_tfds = tf.data.Dataset.from_tensor_slices(popular_articles_tfds)

  # Create a model that takes in raw query features, and
  index = tfrs.layers.factorized_top_k.BruteForce(query_model, k=top_k)

  # recommends only top 10% of products sold in trending days window.
  index.index_from_dataset(
    tf.data.Dataset.zip((top_products_tfds.batch(4096), popular_articles_tfds.batch(4096).map(candidate_model)))
  )

  return index, top_product_ids

def test(index, products_indexed, dataset_name, test_days, customer_columns):
  from signals import get_bought
  test_datastore = Datastore().load_from_dir(dataset_name)

  augment_df_timestamp(test_datastore)
  augment_df_product_sales(test_datastore)
  test_datastore = test_datastore.tail(days=test_days)
  print("test datastore", test_datastore.info())
  test_datastore.transactions = test_datastore.join_all()
  real_bought = get_bought(test_datastore)
  actuals = []

  for customer_id in tqdm(sorted(real_bought), desc="actuals"):
    p = real_bought[customer_id]
    actuals.append(p)

  best_possible_predictions = []
  for customer_id in tqdm(sorted(real_bought), desc="best possible prediction"):
    pset = real_bought[customer_id]
    prediction = []
    for pid in pset:
      if pid in products_indexed:
        prediction.append(pid)
    best_possible_predictions.append(prediction)

  customer_df = test_datastore.transactions[customer_columns].drop_duplicates(keep='first', subset=[Fields.customer_id])
  customer_tfds = dict(customer_df)
  customer_tfds = tf.data.Dataset.from_tensor_slices(customer_tfds).batch(4096)

  predictions = []
  predictions_dict = {}

  for customer in customer_tfds.as_numpy_iterator():
    scores, titles = index(customer)
    customer_ids = customer[Fields.customer_id]
    for cid, score, title in zip(customer_ids, scores.numpy(), titles.numpy()):
      cid = cid.decode("utf-8")
      predictions_dict[cid] = list(title)

  for customer_id in tqdm(sorted(real_bought), desc="predictions"):
    p = predictions_dict[customer_id]
    predictions.append(p)

  from score_submission import mapk, recall
  scores = []
  recalls = []
  best_score = []
  for top_k in [6, 12, 24, 48, 96]:
    score = mapk(actuals, predictions, k=top_k)
    best_score.append(recall(actuals, best_possible_predictions, k=top_k))
    scores.append(score)
    recall_score = recall(actuals, predictions, k=top_k)
    recalls.append(recall_score)
  return scores, recalls, best_score

def save_model(query_model, candidate_model, path):
  query_model.save(path + "/qm/")
  candidate_model.save(path + "/cm/")
  return

def load_model(path):
  qm = tf.keras.models.load_model(path + "/qm/")
  cm = tf.keras.models.load_model(path + "/cm/")
  return qm, cm

def augment_df_timestamp(datastore):
  transactions = datastore.transactions
  ts = transactions[Fields.time_stamp]
  day = ts.dt.day
  month = ts.dt.month
  year = ts.dt.year
  transactions['day'] = day
  transactions['month'] = month
  transactions['year'] = year
  return

def augment_df_product_sales(datastore):
  transactions = datastore.transactions
  transactions[Fields.sales] = 1
  article_groups = transactions[[Fields.sales, Fields.article_id]].groupby([Fields.article_id])
  article_sales = article_groups.cumsum()
  transactions['product_sales_total'] = article_sales

  price_article_groups = transactions[[Fields.price, Fields.article_id]].groupby([Fields.article_id])
  article_avg_prices = price_article_groups.mean()
  article_avg_prices.rename(columns={Fields.price: 'product_price'}, inplace=True)
  transactions = transactions.merge(article_avg_prices, on='article_id')

  timed_article_groups = transactions[[Fields.sales, Fields.article_id, 'year', 'month']].groupby([Fields.article_id, 'year', 'month'])
  timed_article_sales = timed_article_groups.sum()
  transactions = transactions.join(timed_article_sales, on=['article_id', 'year', 'month'], rsuffix='_month')
  transactions.rename(columns={'sales_month': 'product_sales_month'}, inplace=True)
  datastore.transactions = transactions
  category_columns = {'product_sales_month': 5, 'product_sales_total':5, 'product_price':10}
  for column in category_columns:
    q = category_columns[column]
    datastore.transactions[column] = pd.qcut(datastore.transactions[column], q=q, labels=list(range(q)))
    datastore.transactions[column] = datastore.transactions[column].astype(np.int64)


def run():
  dataset_name = "data/full/"
  datastore = Datastore().load_from_dir(dataset_name)

  augment_df_timestamp(datastore)
  augment_df_product_sales(datastore)

  product_feature_columns = ['product_code',
                     'section_no', 'garment_group_no',
                     'product_type_no', 'department_no',
                     'colour_group_name',
                     'index_name']

  product_sales_columns = ['product_sales_month',
                           'product_sales_total',
                           'product_price'
                     ]

  timestamp_columns = ['day', 'month', 'year']

  article_columns = ['article_id'] + product_feature_columns + product_sales_columns

  customer_columns = ['customer_id'] + timestamp_columns

  train_days = 60
  test_days = 8
  trending_days = 15
  epochs = 5
  top_k = 96
  percent_trending = 0.1

  # recommender_model = train(datastore, train_days, test_days, epochs, article_columns, customer_columns)
  # query_model = recommender_model.query_model
  # candidate_model = recommender_model.candidate_model
  # print(query_model.summary())
  # print(candidate_model.summary())
  #
  # save_model(query_model, candidate_model, path="modeldir")
  query_model, candidate_model = load_model(path="modeldir")

  index, products_indexed = model_to_index(query_model, candidate_model, dataset_name, article_columns, test_days, trending_days, top_k, percent_trending)
  scores = test(index, products_indexed, dataset_name, test_days, customer_columns + article_columns)
  print(scores)

  """
  questions : 
  
  WHY IS VAL LOSS INCREASING ? 
  WHAT IS GOING TO TEST IN DATES ? 
  ++ WHY IS MAPK LOWER AT 12 than 6 and so on .... #!%^@$&^@$&^@$^@  : because I was calling the function wrongly and I am stupid
  
  ++ BINNING IS FUCKING BROKEN. replaced hashing with lookups. working now. 
  
  TODO
  0. Tensorboard callback and save configs
  1. ++ capture time by breaking timestamp into mdy
  2. +- capture sales in previous 1,2,3 weeks as columns
  3. don't use article_id embedding for article
  4. can we capture relevant user metrics ? 
  5. cross features
  6. list ranking model
  7. ++ save the query and candidate models separately, and use them to create brute force topk
  8. learning rate reduce before divergence
  """
if __name__ == '__main__':
  run()