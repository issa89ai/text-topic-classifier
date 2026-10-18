# Text Topic Classifier

A supervised machine-learning project comparing TF-IDF text classifiers using
scikit-learn and tracking experiments with MLflow.

## Scope and architecture

Labeled text -> stratified splits -> TF-IDF -> classifier training -> validation
selection -> final test -> saved pipeline and experiment records.

This repository owns its training code, dependencies, tests, CI and local MLflow
artifacts. It needs no other project checkout, shared virtual environment, PDF
assistant, or external model-serving process.

## Task and data

Classify text into four 20 Newsgroups categories: medicine, space, hockey, and
politics/guns. `fetch_20newsgroups` downloads the public dataset on first use and
caches it locally. Headers, footers and quoted replies are removed. The selected
subset has 3,886 examples: 2,331 training, 777 validation and 778 test examples.
Splits are stratified with seed 42. Raw data and trained artifacts are not committed.

## Method

Text -> TF-IDF -> Logistic Regression / Multinomial Naive Bayes / Linear SVM.

- Fit vocabulary and TF-IDF only on training data during model selection.
- Compare six base runs and eight Naive Bayes alpha runs on validation accuracy.
- There are 14 comparison runs but 13 unique configurations: alpha=0.5 repeats.
- Refit the selected configuration on training + validation, then evaluate once
  on the held-out test split (a fifteenth run).
- Record parameters, accuracy, weighted F1 and complete pipelines in MLflow.
- Do not tune against the final test score. Prediction does not update the model.

## Recorded results

The October 15, 2026 run selected **Naive Bayes, alpha=0.1**:

| Measurement | Result |
| --- | ---: |
| Validation accuracy | 90.35% |
| Held-out test accuracy | 92.80% (722 / 778) |
| Held-out test weighted F1 | 92.79% |

These are historical results from the pre-extraction script, not a newly run
benchmark after the move. Older experiments used evaluation data during tuning;
the corpus has been seen before, so this is not a fresh external benchmark.
The extracted script preserves the experiment design, fixes the summary to include
all 14 runs, and explicitly seeds SVM randomness. Dependencies are version-ranged,
not fully locked, so future results may vary.

## Setup and use (PowerShell, Python 3.11 or 3.12)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt -r requirements-dev.txt
python -m pytest -q
python train.py
python -m mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1
```

Run the UI command from this repository. Open http://127.0.0.1:5000.
Training needs internet on the first dataset download. Tests use synthetic text,
need no dataset download, and do not launch training or an MLflow server.
Importing `train.py` does not run experiments. Create the virtual environment in
this repository rather than borrowing another project's installed packages.

## Using a saved model

Copy a model URI from the final run in the local MLflow UI, then load that complete
pipeline with `mlflow.sklearn.load_model(model_uri)` and call
`pipeline.predict(["The satellite entered orbit"])`. Predictions are integer
labels in the order returned by the dataset's `target_names` (alphabetical for
these categories: hockey, medicine, space, politics/guns), not the order of the
input `CATEGORIES` list. Only load model artifacts you trust.

## Local artifacts and limitations

Existing MLflow database and artifacts are local-only. The training script resolves
their locations relative to this repository, not another application's directory.
A fresh clone starts without that private experiment history; run training to
generate your own records. CI runs small regression tests, not the full benchmark.

See [Development record](PROGRESS.md) for historical migration and verification notes.

This is an educational four-topic classifier, not a general-purpose language model
or production classification service. Labels can be ambiguous, vocabulary can drift,
and evaluation on unrelated real-world texts remains future work.
