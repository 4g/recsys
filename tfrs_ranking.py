import pandas as pd
import  tensorflow as tf
import tensorflow_recommenders as tfrs
from embedders import FieldsModel
import numpy as np

class RecommenderModel(tfrs.models.Model):
  def __init__(self, article_model, customer_model, articles, article_feature_names, customer_feature_names):
    super().__init__()
    self.candidate_model = tf.keras.Sequential([
      article_model,
      tf.keras.layers.Dense(32)
    ])
    self.query_model = tf.keras.Sequential([
      customer_model,
      tf.keras.layers.Dense(32)
    ])

    self.article_feature_names = article_feature_names
    self.customer_feature_names = customer_feature_names

    self.task = tfrs.tasks.Retrieval(
        metrics=tfrs.metrics.FactorizedTopK(
            candidates=articles.batch(16384).map(self.candidate_model),
            metrics=[
            tf.keras.metrics.TopKCategoricalAccuracy(
              k=x, name=f"top_{x}") for x in [5, 12]]
        ),
    )

  def compute_loss(self, features, training=False):
    # We only pass the user id and timestamp features into the query model. This
    # is to ensure that the training inputs would have the same keys as the
    # query inputs. Otherwise the discrepancy in input structure would cause an
    # error when loading the query model after saving it.
    query_embeddings = self.query_model({f: features[f] for f in self.customer_feature_names})
    movie_embeddings = self.candidate_model({f: features[f] for f in self.article_feature_names})

    return self.task(query_embeddings, movie_embeddings)

def datastore_to_dataset(datastore):
  train_ds = tf.data.Dataset.from_tensor_slices(dict(datastore.transactions))
  train_ds.shuffle(5_000_000, seed=42, reshuffle_each_iteration=False)
  cached_train = train_ds.prefetch(buffer_size=10_000_000).batch(4096).cache()
  return cached_train

if __name__ == '__main__':
  from datalib import Datastore
  datastore = Datastore().load_from_dir("data/sample42/")
  datastore = datastore.tail(days=15)
  datastore.transactions = datastore.join_all()


  article_columns = ['article_id', 'section_no', 'garment_group_no', 'product_type_no', 'department_no', 'index_name', 't_dat']
  customer_columns = ['customer_id', 't_dat']

  all_columns = list(set(article_columns + customer_columns))
  datastore.transactions = datastore.transactions[all_columns]

  train_datastore, val_datastore = datastore.split_by_date(datastore.end_date - pd.Timedelta(days=7))

  print(train_datastore.info())
  print(val_datastore.info())

  train_datastore.transactions = Datastore.convert_columns_categorical(train_datastore.transactions)
  val_datastore.transactions = Datastore.convert_columns_categorical(val_datastore.transactions)

  df = train_datastore.transactions
  article_model = FieldsModel(train_datastore.transactions, article_columns)
  customer_model = FieldsModel(train_datastore.transactions, customer_columns)

  articles_as_tfds = dict(train_datastore.transactions)
  articles_as_tfds = tf.data.Dataset.from_tensor_slices(articles_as_tfds)
  recommender_model = RecommenderModel(article_model,
                                       customer_model,
                                       articles_as_tfds,
                                       article_columns,
                                       customer_columns)



  train_ds = datastore_to_dataset(train_datastore)
  val_ds = datastore_to_dataset(val_datastore)

  recommender_model.compile(optimizer=tf.keras.optimizers.Adagrad(0.1))
  recommender_model.fit(train_ds, validation_data=val_ds, epochs=100)
  recommender_model.query_model.summary()
  recommender_model.candidate_model.summary()