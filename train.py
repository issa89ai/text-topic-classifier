"""Compare topic classifiers; select on validation and evaluate once on test."""
from pathlib import Path

import mlflow
import mlflow.sklearn
from sklearn.base import clone
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parent
CATEGORIES = ["sci.med", "sci.space", "rec.sport.hockey", "talk.politics.guns"]


def configurations():
    """Preserve the original 14 runs, including the repeated NB alpha=0.5."""
    specs = [
        ("LogisticRegression-C0.1", LogisticRegression(C=0.1, max_iter=1000), 10000, (1, 1)),
        ("LogisticRegression-C1.0", LogisticRegression(C=1.0, max_iter=1000), 10000, (1, 1)),
        ("LogisticRegression-bigrams", LogisticRegression(C=1.0, max_iter=1000), 20000, (1, 2)),
        ("NaiveBayes", MultinomialNB(alpha=0.5), 10000, (1, 1)),
        ("LinearSVC", LinearSVC(C=1.0, max_iter=2000, random_state=42), 10000, (1, 1)),
        ("LinearSVC-bigrams", LinearSVC(C=0.5, max_iter=2000, random_state=42), 20000, (1, 2)),
    ]
    specs.extend(
        (f"NaiveBayes-alpha-{alpha}", MultinomialNB(alpha=alpha), 10000, (1, 1))
        for alpha in [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
    )
    return [(name, Pipeline([
        ("tfidf", TfidfVectorizer(max_features=features, ngram_range=ngrams)),
        ("clf", model),
    ])) for name, model, features, ngrams in specs]


def split_data(texts, labels):
    dev_x, test_x, dev_y, test_y = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    train_x, val_x, train_y, val_y = train_test_split(
        dev_x, dev_y, test_size=0.25, random_state=42, stratify=dev_y
    )
    return train_x, val_x, test_x, train_y, val_y, test_y


def main():
    data = fetch_20newsgroups(
        subset="all", categories=CATEGORIES, remove=("headers", "footers", "quotes")
    )
    train_x, val_x, test_x, train_y, val_y, test_y = split_data(data.data, data.target)
    print(f"Train: {len(train_x)} | Validation: {len(val_x)} | Test: {len(test_x)}")
    mlflow.set_tracking_uri(f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}")
    experiment = mlflow.get_experiment_by_name("doc-topic-classifier")
    if experiment is None:
        mlflow.create_experiment("doc-topic-classifier", artifact_location=(ROOT / "mlruns").as_uri())
    mlflow.set_experiment("doc-topic-classifier")
    results = []
    best_score, best_name, best_pipeline = -1, None, None
    for name, pipeline in configurations():
        with mlflow.start_run(run_name=name):
            pipeline.fit(train_x, train_y)
            predictions = pipeline.predict(val_x)
            accuracy = accuracy_score(val_y, predictions)
            f1 = f1_score(val_y, predictions, average="weighted")
            model = pipeline.named_steps["clf"]
            params = {
                "model": type(model).__name__,
                "max_features": pipeline.named_steps["tfidf"].max_features,
                "ngram_range": str(pipeline.named_steps["tfidf"].ngram_range),
                "split_seed": 42,
            }
            for setting in ("alpha", "C"):
                if hasattr(model, setting):
                    params[setting] = getattr(model, setting)
            mlflow.log_params(params)
            mlflow.set_tag("evaluation_split", "validation")
            mlflow.log_metrics({"accuracy": accuracy, "f1_weighted": f1})
            mlflow.sklearn.log_model(pipeline, name="model")
            results.append((name, accuracy, f1))
            if accuracy > best_score:
                best_score, best_name, best_pipeline = accuracy, name, clone(pipeline)
    print("Validation results (all 14 runs):")
    for name, accuracy, f1 in sorted(results, key=lambda row: row[1], reverse=True):
        print(f"{name:32} accuracy={accuracy:.4f} weighted_f1={f1:.4f}")
    # Recreate the original development-set order for the final refit.
    dev_x, _, dev_y, _ = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42, stratify=data.target
    )
    best_pipeline.fit(dev_x, dev_y)
    with mlflow.start_run(run_name="final-held-out-test"):
        predictions = best_pipeline.predict(test_x)
        metrics = {
            "test_accuracy": accuracy_score(test_y, predictions),
            "test_f1_weighted": f1_score(test_y, predictions, average="weighted"),
        }
        mlflow.log_param("selected_model", best_name)
        mlflow.set_tag("evaluation_split", "test")
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(best_pipeline, name="model")
        print(f"Final test: {best_name}: {metrics}")
    print("14 validation runs + 1 final test run. Do not tune against the test score.")


if __name__ == "__main__":
    main()
