# Data

## Dataset

| Dataset | Source | Classes (label id) | Train | Validation | Test |
| --- | --- | --- | --- | --- | --- |
| TweetEval – sentiment | Hugging Face `cardiffnlp/tweet_eval`, config `sentiment` | 0 negative, 1 neutral, 2 positive | 45,615 | 2,000 | 12,284 |

The dataset is downloaded at runtime by `data.py` and is not stored in this repository.

## Splits

- TweetEval: official splits, unchanged.

## Quality checks

`data.py` checks empty texts, duplicate texts, unexpected labels, class proportions per split and exact text overlap between splits. Run `python data.py` to see the results.

## Citation

- Barbieri et al. (2020), TweetEval. Underlying data: SemEval-2017 Task 4 (Rosenthal et al., 2017).
