import os

from datalib import Datastore, Fields
import numpy as np

import tensorflow as tf
import tensorflow_text as tftext

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

class FieldsModel(tf.keras.Model):
  def __init__(self, articles, text_columns=[], emb_dim=8):
    super().__init__()
    integer_columns = set()
    string_columns = set()

    columns = list(articles.columns)

    for column in columns:
      dtype = articles[column].dtype

      if dtype == np.float32 or dtype == np.float64:
        articles[column].fillna(-1, inplace=True)
        articles[column].astype(np.int64)
        dtype = np.int64

      if column in text_columns:
        continue
      if dtype == np.int64:
        articles[column].fillna(-1, inplace=True)
        integer_columns.add(column)
      else:
        articles[column].fillna("[EMPTY]", inplace=True)
        string_columns.add(column)

    # print(f"text: {text_columns}\ninteger: {integer_columns}\nstring: {string_columns}")

    embedders = {}

    for column in columns:
      if column in integer_columns:
        lookup_layer = tf.keras.layers.IntegerLookup()
        seq_layers = []

      elif column in string_columns:
        lookup_layer = tf.keras.layers.StringLookup()
        seq_layers = [tf.keras.layers.InputLayer(input_shape=[], dtype=tf.string)]

      else:
        continue

      values = np.unique(articles[column].values)
      lookup_layer.adapt(values)
      num_tokens = lookup_layer.vocabulary_size()

      seq_layers += [
        lookup_layer,
        tf.keras.layers.Embedding(num_tokens, emb_dim, mask_zero=True),
      ]
      embedder = tf.keras.Sequential(seq_layers, name=column)
      embedders[column] = embedder

    max_tokens = 10000
    for column in text_columns:
      text_ds = list(articles[column].values)
      text_ds = ["" if x is None else x for x in text_ds]

      # create embedder for text
      text_vectorizer = tf.keras.layers.TextVectorization(
        max_tokens=max_tokens)
      text_vectorizer.adapt(text_ds)
      x = text_vectorizer.vocabulary_size()
      num_tokens = min(x, max_tokens)
      text_embedder = tf.keras.Sequential([
        text_vectorizer,
        tf.keras.layers.Embedding(num_tokens, emb_dim, mask_zero=True),
        tf.keras.layers.GlobalAveragePooling1D(),
      ], name=column)


      embedders[column] = text_embedder

    self.embedders = embedders
    self.integer_columns = list(integer_columns)
    self.text_columns = list(text_columns)
    self.string_columns = list(string_columns)

  def call(self, article):
    values = []
    for column in self.embedders:
      embedder = self.embedders[column]
      # print(embedder.summary())
      value = article[column]
      # if (column in self.string_columns) or (column in self.text_columns):
      #   value = tf.constant([value])
      embedding = embedder(value)
      values.append(embedding)

    return tf.concat(values, axis=1)

if __name__ == '__main__':
  datastore = Datastore().load_from_dir("data/sample42/")

  article_model = FieldsModel(datastore.articles, {'detail_desc', 'prod_name'})
  customer_model = FieldsModel(datastore.customers)

  article = datastore.articles.sample(1).to_dict(orient="records")[0]
  customer = datastore.customers.sample(1).to_dict(orient="records")[0]

  embedding = article_model(article)
  print(embedding)

  embedding = customer_model(customer)
  print(embedding)

