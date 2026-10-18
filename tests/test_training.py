import importlib.util
from pathlib import Path

from sklearn.base import clone

spec = importlib.util.spec_from_file_location("classifier_train", Path(__file__).parents[1] / "train.py")
train = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train)


def test_fourteen_runs_and_one_duplicate():
    configs = train.configurations()
    assert len(configs) == 14
    assert len({name for name, _ in configs}) == 14
    signatures = {
        (type(p.named_steps["clf"]).__name__, str(p.named_steps["clf"].get_params()),
         p.named_steps["tfidf"].max_features, p.named_steps["tfidf"].ngram_range)
        for _, p in configs
    }
    assert len(signatures) == 13


def test_split_is_disjoint_stratified_and_reproducible():
    texts = [f"example {i}" for i in range(100)]
    labels = [i % 4 for i in range(100)]
    result = train.split_data(texts, labels)
    assert result == train.split_data(texts, labels)
    x, v, t, y, vy, ty = result
    assert [len(x), len(v), len(t)] == [60, 20, 20]
    assert not (set(x) & set(v) or set(x) & set(t) or set(v) & set(t))
    assert set(x + v + t) == set(texts)
    assert all(set(group) == {0, 1, 2, 3} for group in (y, vy, ty))


def test_every_pipeline_can_learn_and_predict_without_downloading():
    texts = ["orbit satellite space", "planet rocket space", "hockey puck goal", "team hockey score"]
    for _, pipeline in train.configurations():
        model = clone(pipeline).fit(texts, [0, 0, 1, 1])
        assert list(model.predict(["satellite orbit", "hockey goal"])) == [0, 1]
        assert "unseenword" not in model.named_steps["tfidf"].vocabulary_
