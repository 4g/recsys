from lightgbm.sklearn import LGBMRanker
from sklearn.model_selection import train_test_split

import pandas as pd
from datalib import Datastore, Fields

datastore = Datastore()
datastore.load_from_dir("data/sample31")

# data = data.tail(days=365)

datastore.join_all()
from signals import augment_df_timestamp
augment_df_timestamp(datastore)

print(datastore.transactions)

data = datastore.transactions

c = list(data.columns)

categorical_columns = [Fields.time_stamp, Fields.customer_id]

data[categorical_columns] = (
                   data[categorical_columns]
                   .apply(lambda x: pd.factorize(x)[0])).astype('int8')

import numpy as np
column = 'price'
data[column] = pd.qcut(data[column], q=10, labels=list(range(10)))
data[column] = data[column].astype(np.int64)

X = data[c]
X.sort_values([Fields.customer_id, 'week'], inplace=True)
y = data[Fields.price]

query_train = X_train.groupby([Fields.customer_id, 'week'])[Fields.article_id].count().values
query_test = X_test.groupby([Fields.customer_id, 'week'])[Fields.article_id].count().values

gbm = LGBMRanker(
    objective="lambdarank",
    metric="ndcg",
    boosting_type="dart",
    max_depth=7,
    n_estimators=30,
    importance_type='gain',
    verbose=10
)

X_train.drop(columns = ['t_dat', 'customer_id', 'article_id', 'price'])


gbm.fit(X_train, y_train, group=query_train, eval_group=[query_test])

test_pred = gbm.predict(X_test)

X_test["predicted_ranking"] = test_pred
X_test.sort_values("predicted_ranking", ascending=False, inplace=True)

print(X_test["predicted_ranking"].sum())
print(X_test)