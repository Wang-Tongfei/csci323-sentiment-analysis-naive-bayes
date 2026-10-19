# Data

Owner: C (Data). Fill in every TBD once the data is downloaded.

## Datasets

| Dataset | Source | Classes (label id) | Train | Validation | Test | In git? |
| --- | --- | --- | --- | --- | --- | --- |
| TweetEval – sentiment | Hugging Face `cardiffnlp/tweet_eval`, config `sentiment` | 0 negative, 1 neutral, 2 positive | TBD | TBD | TBD | Yes (small) |
| IMDb | Hugging Face `stanfordnlp/imdb` | 0 negative, 1 positive | TBD | TBD (our split) | TBD | No (too large); split indices only |

## Splits

- TweetEval: official splits, unchanged.
- IMDb: stratified TBD% of the official train split held out as validation; random seed TBD.

## Quality checks

| Check | TweetEval | IMDb |
| --- | --- | --- |
| Empty texts | TBD | TBD |
| Duplicate texts | TBD | TBD |
| Train/test overlap (leakage) | TBD | TBD |
| Class proportions per split | TBD | TBD |

## Citation and licence

- Barbieri et al. (2020), TweetEval. Underlying data: SemEval-2017 Task 4 (Rosenthal et al., 2017). Licence/terms: TBD — check the dataset card.
- Maas et al. (2011), IMDb Large Movie Review Dataset. Licence/terms: TBD — check the dataset card.
