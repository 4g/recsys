import os

from datalib import Datastore, Fields
import numpy as np

import tensorflow as tf
import tensorflow_text as tftext

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

class FieldsModel(tf.keras.Model):
  def __init__(self, features, columns, text_columns=(), emb_dim=8):
    super().__init__()

    embedders = {}

    for column in columns:
      lookup_layer = tf.keras.layers.StringLookup()
      seq_layers = [tf.keras.layers.InputLayer(input_shape=[], dtype=tf.string)]

      values = np.unique(features[column].values)
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
      text_ds = list(features[column].values)
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

  def call(self, article):
    values = []
    for column in self.embedders:
      embedder = self.embedders[column]
      value = article[column]
      embedding = embedder(value)
      values.append(embedding)

    return tf.concat(values, axis=1)

if __name__ == '__main__':
  datastore = Datastore().load_from_dir("data/sample42/")
  datastore.transactions = datastore.join_all()

  article_columns = ["article_id", 'section_no', 'garment_group_no', 'product_type_no', 't_dat']

  datastore.convert_columns_categorical(datastore.transactions)
  article_model = FieldsModel(datastore.transactions,
                              columns=article_columns)

  article = datastore.transactions[article_columns]
  train_ds = tf.data.Dataset.from_tensor_slices(dict(article))
  train_ds.shuffle(10, seed=42, reshuffle_each_iteration=False)
  cached_train = train_ds.prefetch(buffer_size=10).batch(10).cache()

  from tqdm import tqdm
  for elem in tqdm(cached_train.take(1000)):
    embedding = article_model(elem)



