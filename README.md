# Sentiment Analysis with a Naive Bayes Classifier

CSCI323 Modern AI — Group Project, Topic 4.3 (University of Wollongong)

We build a text classification system for sentiment analysis around a **hand-written multinomial Naive Bayes classifier**, tune it with a proper train / validation / test workflow, and evaluate systematically where it works and where it fails.

- **Main dataset:** [TweetEval – sentiment](https://huggingface.co/datasets/cardiffnlp/tweet_eval) (negative / neutral / positive, official splits)
- **Second dataset (optional):** [IMDb](https://huggingface.co/datasets/stanfordnlp/imdb) (negative / positive)
- **Comparison models:** majority class, logistic regression, linear SVM; deep learning (DistilBERT / twitter-roberta) as a bonus

> **Golden rule:** hyperparameters are chosen on the validation set only. The test set is used exactly once, for the final evaluation.

## Team

| Role | Leads | Code area | Member |
| --- | --- | --- | --- |
| A Introduction | §1 Introduction | — | TBD |
| B Theory | §2 Background Theory | — | TBD |
| C Data | §3a Data Pipeline & Baseline | Data loading, preprocessing, features | TBD |
| D Algorithm | §3b NB from Scratch & Comparison | Naive Bayes implementation, baselines, deep learning (bonus) | TBD |
| E Evaluation | §4 Evaluation & Results Analysis | Tuning, evaluation, figures | TBD |
| F Conclusion | §5 Conclusion & References | — | TBD |

## Repository layout (planned)

> Planned structure only — subject to change. Each module owner decides the final file names and layout.

```
├── data/              # datasets (small ones committed), split indices, data README    (C)
├── scripts/           # data download script                                           (C)
├── data.py            # module a: loading, splitting, preprocessing                    (C)
├── features.py        # tokenising, n-grams, count / binary / TF-IDF                   (C)
├── naive_bayes.py     # module b: hand-written multinomial (and Bernoulli) NB          (D)
├── baselines.py       # majority class, logistic regression, linear SVM, NB-SVM        (D)
├── tune.py            # grid search on the validation set                              (E)
├── evaluate.py        # module c: metrics, confusion matrix, learning curves, ablations (E)
├── deep/              # optional bonus: Transformer fine-tuning notebook (Colab)       (D)
├── tests/             # sanity checks, e.g. our NB matches sklearn                     (D)
└── report/figures/    # all figures used in the report and slides                      (E)
```

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## How to run

To be added by the module owners.

## Working together

- Everyone commits from **their own GitHub account**; the commit history is evidence for the contribution table.
- Work on your own branch and open a pull request; merge it into `main` once GitHub shows no conflicts.
- Do not commit large raw data (IMDb), virtual environments or model checkpoints; see `.gitignore`.

## Academic integrity

Any use of generative AI tools is declared in the report following UOW referencing guidelines.

## References

- Barbieri, F., Camacho-Collados, J., Espinosa Anke, L., & Neves, L. (2020). TweetEval: Unified benchmark and comparative evaluation for tweet classification. *Findings of the Association for Computational Linguistics: EMNLP 2020*, 1644–1650.
- Rosenthal, S., Farra, N., & Nakov, P. (2017). SemEval-2017 Task 4: Sentiment analysis in Twitter. *Proceedings of the 11th International Workshop on Semantic Evaluation (SemEval-2017)*, 502–518.
- Maas, A. L., Daly, R. E., Pham, P. T., Huang, D., Ng, A. Y., & Potts, C. (2011). Learning word vectors for sentiment analysis. *Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics*, 142–150.
