from datalib import Datastore, Fields
import pandas as pd
import random
import numpy as np

import tensorflow_recommenders as tfrs
import tensorflow as tf

from tqdm import tqdm

from typing import Dict, Text


def train():
  slice_days = 22
  num_test_days = 7
  data = Datastore().load_from_dir("data/sample42/")
  data.transactions.drop_duplicates(keep='first', inplace=True)
  slice_data = data.get_random_slice(slice_days + num_test_days + 1)
  data_train, data_test = slice_data.split_by_date(slice_data.end_date - pd.Timedelta(days=num_test_days + 1))
  print(data_train.info(), data_test.info())

  unique_user_ids = np.unique(list(data_train.transactions.customer_id.values))
  unique_articles = np.unique(list(data_train.transactions.article_id.values))
  articles_tfds = tf.data.Dataset.from_tensor_slices(unique_articles)

  from signals import get_bought, get_repeat_counts
  last_15_days = data_train.tail(days=15)
  product_counts = get_repeat_counts(last_15_days.transactions, Fields.article_id)

  k = .1
  sorted_products = sorted(product_counts, key=product_counts.get, reverse=True)
  top_products = sorted_products[:int(len(sorted_products)*k)]
  # print(top_products)
  top_products = tf.data.Dataset.from_tensor_slices(list(top_products))

  model = RecommenderModel(unique_user_ids, unique_articles)
  model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))


  tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="logs", histogram_freq=1)
  callbacks = [tensorboard_callback]


  train_ds = tf.data.Dataset.from_tensor_slices(dict(data_train.transactions[['customer_id', 'article_id']]))
  tf.random.set_seed(42)
  train_ds.shuffle(5_000_000, seed=42, reshuffle_each_iteration=False)
  cached_train = train_ds.prefetch(buffer_size=10_000_000).batch(4096).cache()

  test_ds = tf.data.Dataset.from_tensor_slices(dict(data_test.transactions[['customer_id', 'article_id']]))
  test_ds.shuffle(1000_000, seed=42, reshuffle_each_iteration=False)
  cached_test = test_ds.batch(4096).cache()


  model.fit(cached_train, validation_data=cached_test, epochs=8, callbacks=callbacks)

  results = model.evaluate(cached_test, return_dict=True)
  print(results)


  # Create a model that takes in raw query features, and
  index = tfrs.layers.factorized_top_k.BruteForce(model.user_model, k=12)
  # recommends movies out of the entire movies dataset.
  index.index_from_dataset(
    tf.data.Dataset.zip((top_products.batch(100), top_products.batch(100).map(model.movie_model)))
  )

  real_bought = get_bought(data_test)
  actuals = []
  predictions = []

  for customer_id in tqdm(sorted(real_bought), desc="actuals"):
    p = real_bought[customer_id]
    actuals.append(p)

  for customer_id in tqdm(sorted(real_bought), desc="predictions"):
    _, titles = index(tf.constant([customer_id]))
    predictions.append(list(titles[0]))

  from score_submission import mapk
  score = mapk(predictions, actuals, k=12)
  print(score)

train()