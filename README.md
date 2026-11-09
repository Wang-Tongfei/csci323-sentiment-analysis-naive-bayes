# Sentiment Analysis with a Naive Bayes Classifier

**CSCI323 Modern Artificial Intelligence — Group Project, Topic 4.3**  
University of Wollongong

This project develops a sentiment classification system centred around a **Multinomial Naive Bayes classifier implemented from scratch using NumPy**.

The system uses the **TweetEval sentiment dataset** to classify tweets as **Negative, Neutral or Positive**. Three machine-learning classifiers are implemented and compared:

- Custom Multinomial Naive Bayes
- Logistic Regression
- Linear SVM

The project follows a train / validation / test workflow.

> **Golden rule:** Hyperparameters are selected using the validation set only. The test set is reserved for the final evaluation.

---

## Team Contributions

| Member | Part | Responsibility | Main Contribution |
| --- | --- | --- | --- |
| **[Name]** | Part 1 | Introduction | Problem background, motivation, project objectives and overview |
| **[Name]** | Part 2 | Background Theory | Theoretical foundations, mathematical concepts and related research |
| **Emma Louise McLean/ Mohammed Omar** | Part 3A | Data Pipeline & Features | Dataset loading, data checks and Bag-of-Words feature extraction |
| **Emma Louise McLean/ Mohammed Omar** | Part 3B | Model Development | Custom Naive Bayes, Logistic Regression, Linear SVM and model verification |
| **Ng Xuan Ying** | Part 4 | Evaluation & Results | Hyperparameter tuning, metrics, ablation study, learning curve, figures and error analysis |
| **[Name]** | Part 5 | Conclusion & References | Main findings, limitations, future improvements and references |

All members also contributed to reviewing and integrating the final project, report and presentation.

---

# Part 1 — Introduction

Sentiment analysis is a text-classification task that aims to determine the opinion or emotional polarity expressed in text.

In this project, the task is to classify tweets into three sentiment categories:

- Negative
- Neutral
- Positive

The main objective is to implement a **Multinomial Naive Bayes classifier from scratch** and investigate how well it performs compared with other classical machine-learning methods.

The project compares:

1. Custom Multinomial Naive Bayes
2. Logistic Regression
3. Linear SVM

The project also investigates how feature representation, hyperparameter selection and training-data size affect model performance.

---

# Part 2 — Background Theory

The project is based on several concepts from machine learning and text classification.

## Bag-of-Words

Machine-learning models cannot directly process raw sentences, so tweets are converted into numerical features.

The main system uses **Bag-of-Words**, where each feature represents how many times a vocabulary word appears in a tweet.

The baseline system uses **unigrams**, meaning each individual word is treated as one feature.

---

## Multinomial Naive Bayes

Multinomial Naive Bayes is a probability-based classifier commonly used for text classification.

The model learns:

- how common each sentiment class is;
- how frequently each word appears within each sentiment class.

The classifier then combines this information to determine which sentiment class is most likely for a new tweet.

The implementation also uses **additive smoothing**, controlled by the hyperparameter `alpha`, to prevent unseen words from producing zero probabilities.

---

## Logistic Regression

Logistic Regression is used as a comparison model.

It learns weights for the Bag-of-Words features and uses these weights to classify the sentiment of a tweet.

Its main tuned hyperparameter is `C`, which controls the strength of regularisation.

---

## Linear SVM

Linear Support Vector Machine (Linear SVM) is the second comparison model.

It learns a decision boundary that separates the sentiment classes while attempting to maintain a suitable margin between them.

Its main tuned hyperparameter is also `C`.

---

## Research Connection

The project is related to earlier research showing that classical machine-learning methods can be effective for sentiment classification.

A key reference is:

**Pang, Lee and Vaithyanathan (2002), _Thumbs up? Sentiment Classification using Machine Learning Techniques_.**

The TweetEval benchmark is based on:

**Barbieri et al. (2020), _TweetEval: Unified Benchmark and Comparative Evaluation for Tweet Classification_.**

---

# Part 3 — System Development

## Part 3A — Data Pipeline and Feature Representation

### Dataset

The project uses:

[TweetEval — Sentiment](https://huggingface.co/datasets/cardiffnlp/tweet_eval)

TweetEval provides three sentiment labels:

| Label | Sentiment |
| --- | --- |
| `0` | Negative |
| `1` | Neutral |
| `2` | Positive |

The official dataset splits are retained:

| Split | Samples | Purpose |
| --- | ---: | --- |
| Training | 45,615 | Train the models |
| Validation | 2,000 | Select model settings |
| Test | 12,284 | Final evaluation |

---

### Data Checks

`data.py` loads and checks the TweetEval dataset.

Checks include:

- expected train, validation and test splits;
- required text and label columns;
- empty tweets;
- duplicate tweet text;
- unexpected labels;
- class distribution;
- exact tweet overlap between dataset splits.

---

### Feature Extraction

`features.py` converts tweets into numerical **Bag-of-Words** features using `CountVectorizer`.

The main configuration is:

```python
CountVectorizer(
    lowercase=True,
    min_df=1
)
```

The default:

```python
ngram_range=(1, 1)
```

means that the main pipeline uses **unigrams only**.

The vocabulary is learned from the training data:

```python
X_train = vectorizer.fit_transform(train_texts)
```

The same vocabulary is then applied to the validation and test data:

```python
X_validation = vectorizer.transform(validation_texts)
X_test = vectorizer.transform(test_texts)
```

This helps prevent the validation or test data from influencing the feature vocabulary.

The final baseline vocabulary contains:

**43,358 features**

Feature matrices:

```text
X_train: 45,615 × 43,358
X_val:    2,000 × 43,358
X_test:  12,284 × 43,358
```

---

## Part 3B — Model Development

The same unigram Bag-of-Words features are provided to all three main classifiers.

### Custom Multinomial Naive Bayes

The main classifier is implemented from scratch using NumPy in:

```text
naive_bayes.py
```

The model learns:

- class probabilities;
- word probabilities for each sentiment class;
- additive smoothing using `alpha`;
- log-probability scores for prediction.

---

### Logistic Regression

Logistic Regression is implemented using scikit-learn in:

```text
baselines.py
```

Its main tuned parameter is:

```text
C
```

---

### Linear SVM

Linear SVM is also implemented in:

```text
baselines.py
```

using:

```python
LinearSVC(loss="hinge")
```

Its main tuned parameter is also:

```text
C
```

---

### Custom Naive Bayes Verification

The custom Naive Bayes implementation is compared against scikit-learn's `MultinomialNB`.

Both models use:

```text
alpha = 1.0
```

The comparison produced:

- identical validation metrics;
- 100% prediction agreement;
- only very small floating-point differences.

The scikit-learn Naive Bayes model is used only as a **correctness check** and is not counted as a fourth main method.

---

# Part 4 — Evaluation and Results

## Hyperparameter Tuning

Hyperparameters are selected using the **validation set only**.

The model-selection metric is:

**Macro Recall**

### Naive Bayes

Tested:

```text
alpha = 0.1, 0.5, 1.0, 2.0
```

Selected:

```text
alpha = 0.5
```

### Logistic Regression

Tested:

```text
C = 0.01, 0.1, 1.0, 10.0
```

Selected:

```text
C = 1.0
```

### Linear SVM

Tested:

```text
C = 0.001, 0.01, 0.1, 1.0
```

Selected:

```text
C = 0.1
```

---

## Final Test Results

The models are evaluated using:

- Accuracy
- Macro Recall
- Macro F1

| Model | Accuracy | Macro Recall | Macro F1 |
| --- | ---: | ---: | ---: |
| Custom Multinomial Naive Bayes | 0.5797 | **0.5915** | **0.5771** |
| Logistic Regression | 0.5817 | 0.5648 | 0.5623 |
| Linear SVM | **0.5863** | 0.5491 | 0.5469 |

The results show that there is not one model that is best for every metric.

- **Linear SVM** achieved the highest test Accuracy.
- **Custom Multinomial Naive Bayes** achieved the highest Macro Recall and Macro F1.

---

## Feature Ablation Study

A separate feature experiment investigates whether changing the Bag-of-Words representation affects Naive Bayes performance.

The same custom Naive Bayes model is used with:

```text
alpha = 0.5
```

and evaluated on the validation set.

| Feature Configuration | Vocabulary Size | Validation Macro Recall |
| --- | ---: | ---: |
| Original unigrams (`min_df=1`) | 43,358 | 0.5879 |
| Unigrams + bigrams | 387,765 | 0.5533 |
| Unigrams (`min_df=2`) | 19,120 | **0.5955** |

The experiment showed that adding bigrams did not improve Naive Bayes validation performance.

Removing words that appeared in fewer than two training tweets produced a smaller feature space and slightly higher validation Macro Recall.

> The main three-model comparison still uses the original unigram Bag-of-Words representation.

---

## Learning Curve

The project also evaluates Naive Bayes using different amounts of training data:

| Training Data Used | Validation Macro Recall |
| ---: | ---: |
| 20% | 0.5090 |
| 40% | 0.5453 |
| 60% | 0.5664 |
| 80% | 0.5857 |
| 100% | 0.5879 |

Performance generally improves as more training data is used.

---

## Model Interpretation

`top_words.py` examines words that the Naive Bayes classifier learned as being particularly distinctive for each sentiment class.

The results are stored in:

```text
results/top_weighted_words.csv
```

---

## Error Analysis

The project also analyses incorrect predictions to understand where the models fail.

Evaluation includes:

- Naive Bayes confusion matrix;
- comparison between the three models;
- Naive Bayes alpha-tuning graph;
- sampled misclassified tweets;
- sentiment error categories;
- comparison of errors made by Naive Bayes, Logistic Regression and Linear SVM.

Main files:

```text
evaluate.py
prediction_error_analysis.py
```

---

# Part 5 — Conclusion

The project successfully implemented a **Multinomial Naive Bayes classifier from scratch** and compared it with Logistic Regression and Linear SVM on the TweetEval sentiment dataset.

The final results demonstrate that model performance depends on the evaluation metric.

Linear SVM achieved the highest overall test Accuracy, while the custom Naive Bayes classifier achieved the highest Macro Recall and Macro F1.

The custom Naive Bayes implementation also produced the same predictions as scikit-learn's implementation during the correctness check, supporting the correctness of the implementation.

The additional experiments showed that:

- increasing the amount of training data generally improved Naive Bayes validation performance;
- adding bigram features did not improve the Naive Bayes model in the ablation experiment;
- removing extremely rare word features produced a small improvement in validation Macro Recall.

A limitation of the main system is that the baseline Bag-of-Words representation uses individual words and therefore does not directly represent the full order and context of language.

Possible future improvements include:

- further feature-selection experiments;
- alternative text representations;
- additional n-gram configurations;
- further hyperparameter investigation;
- neural or transformer-based sentiment models.

---

# Repository Structure

```text
├── data.py
│   └── TweetEval loading and data-quality checks
│
├── features.py
│   └── Unigram Bag-of-Words feature extraction
│
├── naive_bayes.py
│   └── Custom Multinomial Naive Bayes
│
├── baselines.py
│   ├── sklearn Naive Bayes reference
│   ├── Logistic Regression
│   └── Linear SVM
│
├── run_experiments.py
│   ├── Custom NB verification
│   ├── Hyperparameter tuning
│   └── Final test evaluation
│
├── ablation_study.py
│   └── Feature ablation experiments
│
├── learning_curve.py
│   └── Training-size experiment
│
├── top_words.py
│   └── Naive Bayes word analysis
│
├── evaluate.py
│   └── Evaluation figures and error analysis
│
├── prediction_error_analysis.py
│   └── Detailed prediction-error analysis
│
├── test/
│   ├── test.py
│   └── test_pipeline.py
│
├── results/
│   └── Experiment results and predictions
│
├── report/
│   └── figures/
│
├── deep/
│   └── Reserved for optional future deep-learning work
│
├── requirements.txt
└── README.md
```

---

# Setup

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the environment.

Windows:

```bash
.venv\Scripts\activate
```

macOS / Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Current requirements:

```text
numpy
pandas
scikit-learn
matplotlib
datasets
```

---

# How to Run

## Main Experiment

```bash
python run_experiments.py
```

This runs:

1. feature loading;
2. custom Naive Bayes verification;
3. validation tuning for all three classifiers;
4. final test evaluation;
5. result saving.

Main outputs:

```text
results/validation_results.csv
results/test_predictions.csv
results/summary.json
```

---

## Additional Experiments

Feature ablation:

```bash
python ablation_study.py
```

Learning curve:

```bash
python learning_curve.py
```

Top-word analysis:

```bash
python top_words.py
```

Evaluation and figures:

```bash
python evaluate.py
```

Prediction error analysis:

```bash
python prediction_error_analysis.py
```

Pipeline tests:

```bash
python test/test_pipeline.py
```

---

# Project Pipeline

```text
TweetEval labelled tweets
        ↓
Data quality checks
        ↓
Unigram Bag-of-Words
        ↓
43,358 numerical word features
        ↓
Custom Naive Bayes
Logistic Regression
Linear SVM
        ↓
Validation-based hyperparameter selection
        ↓
NB alpha = 0.5
LR C = 1.0
SVM C = 0.1
        ↓
Final test evaluation
        ↓
Accuracy / Macro Recall / Macro F1
        ↓
Additional analysis and error investigation
```

---

# Working Together

- Each member commits from their own GitHub account.
- Members work on their own branch before merging completed work into `main`.
- Pull requests should be checked before merging.
- Do not commit virtual environments, unnecessary large files or model checkpoints.
- Experimental results used in the report should be reproducible from the repository.

---

# Academic Integrity

Any use of generative AI tools is declared in the final report according to University of Wollongong academic-integrity and referencing requirements.

---

# References

Barbieri, F., Camacho-Collados, J., Espinosa-Anke, L., & Neves, L. (2020).  
**TweetEval: Unified Benchmark and Comparative Evaluation for Tweet Classification.** Findings of the Association for Computational Linguistics: EMNLP 2020.

Pang, B., Lee, L., & Vaithyanathan, S. (2002).  
**Thumbs up? Sentiment Classification using Machine Learning Techniques.** Proceedings of EMNLP 2002.
