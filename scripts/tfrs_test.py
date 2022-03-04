from typing import Dict, Text

import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_recommenders as tfrs

# Ratings data.
ratings = tfds.load('movielens/1m-ratings', split="train")
# Features of all the available movies.
movies = tfds.load('movielens/1m-movies', split="train")

# Select the basic features.
ratings = ratings.map(lambda x: {
  "movie_id": tf.strings.to_number(x["movie_id"]),
  "user_id": tf.strings.to_number(x["user_id"])
})
movies = movies.map(lambda x: tf.strings.to_number(x["movie_id"]))


# Build a model.
class Model(tfrs.Model):

  def __init__(self):
    super().__init__()

    # Set up user representation.
    self.user_model = tf.keras.layers.Embedding(
      input_dim=7000, output_dim=64)
    # Set up movie representation.
    self.item_model = tf.keras.layers.Embedding(
      input_dim=4000, output_dim=64)
    # Set up a retrieval task and evaluation metrics over the
    # entire dataset of candidates.
    top_k = 12
    self.task = tfrs.tasks.Retrieval(
      metrics=tfrs.metrics.FactorizedTopK(
        candidates=movies.batch(128).map(self.item_model),
        metrics=[tf.keras.metrics.TopKCategoricalAccuracy(k=top_k, name=f"top_{top_k}_categorical_accuracy")],
        k=12,
        name="met"
      )
    )

  def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
    user_embeddings = self.user_model(features["user_id"])
    movie_embeddings = self.item_model(features["movie_id"])

    return self.task(user_embeddings, movie_embeddings)

  def __call__(self, *args, **kwargs):
    pass


model = Model()
model.compile(optimizer=tf.keras.optimizers.Adagrad(0.5))

# Randomly shuffle data and split between train and test.
tf.random.set_seed(42)
shuffled = ratings.shuffle(1000_000, seed=42, reshuffle_each_iteration=False)

train = shuffled.take(800_000)
test = shuffled.skip(800_000).take(200_000)

# Train.
model.fit(train.batch(4096), epochs=1)

model.save_weights("tfrs_model.hdf5")

model2 = Model()

model2.load_weights("tfrs_model.hdf5")
# Evaluate.
model2.evaluate(test.batch(4096), return_dict=True)
