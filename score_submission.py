from tqdm import tqdm
import numpy as np

def apk(actual, predicted, k=12):
    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
      if p in actual and p not in predicted[:i]:
          num_hits += 1.0
          score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)

def mapk(actual, predicted, k=12):
    return np.mean([apk(a,p,k) for a,p in zip(actual, predicted)])

def recall(actual, predicted, k=12):
  score = 0
  for a, p in zip(actual, predicted):
    seta = set(a)
    setp = set(p[:k])
    score += len(seta.intersection(setp)) / len(seta)

  score = score / len(actual)
  return score


def get_score(predicted, real):
  empty_set = set()
  prec_sum = 0
  n_customers = 0
  total_correct = 1
  total_possible = 1
  actuals = []
  predictions = []
  for customer_id, prediction in predicted.items():
    actual = real.get(customer_id, empty_set)
    if actual:
      actuals.append(actual)
      predictions.append(prediction)
      # print(actual, prediction)

  print(len(actuals), len(predictions))
  prec_sum = mapk(actuals, predictions, k=12)

  print(f"Scores: {total_correct}/{total_possible}={total_correct/total_possible}, {prec_sum}")
  return prec_sum

def score_submission(f_submission, f_transactions):
  predicted_buys = {}
  real_buys = {}
  skip_header = True

  for line in tqdm(open(f_submission)):
    if skip_header:
      skip_header = False
      continue
    customer_id, product_ids = line.strip().split(",")
    predicted_buys[customer_id] = product_ids.split(" ")

  skip_header = True
  for line in tqdm(open(f_transactions)):
    if skip_header:
      skip_header = False
      continue
    date, customer_id, product_id, price, channel = line.strip().split(",")
    real_buys[customer_id] = real_buys.get(customer_id, set())
    real_buys[customer_id].add(product_id)


  get_score(predicted_buys, real_buys)

if __name__ == "__main__":
  import argparse

  parser = argparse.ArgumentParser()
  parser.add_argument("--submission", default=None, help="Load datastore input dir", required=True)
  parser.add_argument("--transactions", default=None, help="Output recommendations file", required=True)

  args = parser.parse_args()
  score_submission(args.submission, args.transactions)
