from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


EPS = 1e-12


class LabelEncoder:
    def __init__(self) -> None:
        self.classes_: list[str] = []
        self.lookup_: dict[str, int] = {}

    def fit(self, labels: Iterable[str]) -> "LabelEncoder":
        self.classes_ = sorted(set(str(label) for label in labels))
        self.lookup_ = {label: idx for idx, label in enumerate(self.classes_)}
        return self

    def transform(self, labels: Iterable[str]) -> np.ndarray:
        return np.asarray([self.lookup_[str(label)] for label in labels], dtype=int)

    def inverse_transform(self, y: Iterable[int]) -> list[str]:
        return [self.classes_[int(idx)] for idx in y]


class Preprocessor:
    def __init__(self) -> None:
        self.median_: np.ndarray | None = None
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> "Preprocessor":
        finite = np.where(np.isfinite(x), x, np.nan)
        median = np.nanmedian(finite, axis=0)
        median = np.where(np.isfinite(median), median, 0.0)
        cleaned = np.where(np.isfinite(x), x, median)
        mean = np.mean(cleaned, axis=0)
        std = np.std(cleaned, axis=0)
        std = np.where(std > EPS, std, 1.0)
        self.median_ = median
        self.mean_ = mean
        self.std_ = std
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.median_ is None or self.mean_ is None or self.std_ is None:
            raise RuntimeError("Preprocessor must be fit before transform")
        cleaned = np.where(np.isfinite(x), x, self.median_)
        return (cleaned - self.mean_) / self.std_

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return self.fit(x).transform(x)


def stratified_split(
    y: np.ndarray,
    seed: int,
    train_fraction: float = 0.70,
    val_fraction: float = 0.15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train: list[int] = []
    val: list[int] = []
    test: list[int] = []
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        n_train = max(1, int(round(len(idx) * train_fraction)))
        n_val = max(1, int(round(len(idx) * val_fraction))) if len(idx) >= 4 else 0
        n_train = min(n_train, len(idx) - n_val)
        train.extend(idx[:n_train])
        val.extend(idx[n_train : n_train + n_val])
        test.extend(idx[n_train + n_val :])

    for bucket in (train, val, test):
        rng.shuffle(bucket)
    return np.asarray(train, dtype=int), np.asarray(val, dtype=int), np.asarray(test, dtype=int)


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred)) if y_true.size else 0.0


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    matrix = np.zeros((n_classes, n_classes), dtype=int)
    for actual, pred in zip(y_true, y_pred):
        matrix[int(actual), int(pred)] += 1
    return matrix


def macro_f1_score(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    scores = []
    for cls in range(n_classes):
        tp = float(np.sum((y_true == cls) & (y_pred == cls)))
        fp = float(np.sum((y_true != cls) & (y_pred == cls)))
        fn = float(np.sum((y_true == cls) & (y_pred != cls)))
        precision = tp / max(tp + fp, EPS)
        recall = tp / max(tp + fn, EPS)
        scores.append((2.0 * precision * recall) / max(precision + recall, EPS))
    return float(np.mean(scores))


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.any(mask):
        return {"mae": float("nan"), "rmse": float("nan"), "r2": float("nan")}
    actual = y_true[mask]
    pred = y_pred[mask]
    err = pred - actual
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    denom = float(np.sum((actual - np.mean(actual)) ** 2))
    r2 = 1.0 - float(np.sum(err**2)) / max(denom, EPS)
    return {"mae": mae, "rmse": rmse, "r2": r2}


class GaussianNB:
    def __init__(self) -> None:
        self.classes_: np.ndarray | None = None
        self.mean_: np.ndarray | None = None
        self.var_: np.ndarray | None = None
        self.prior_: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "GaussianNB":
        classes = np.unique(y)
        means = []
        vars_ = []
        priors = []
        for cls in classes:
            subset = x[y == cls]
            means.append(np.mean(subset, axis=0))
            vars_.append(np.var(subset, axis=0) + 1e-6)
            priors.append(subset.shape[0] / x.shape[0])
        self.classes_ = classes
        self.mean_ = np.vstack(means)
        self.var_ = np.vstack(vars_)
        self.prior_ = np.asarray(priors)
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.var_ is None or self.prior_ is None or self.classes_ is None:
            raise RuntimeError("GaussianNB must be fit before predict")
        log_probs = []
        for idx in range(len(self.classes_)):
            log_prior = np.log(self.prior_[idx] + EPS)
            log_likelihood = -0.5 * np.sum(
                np.log(2.0 * np.pi * self.var_[idx]) + ((x - self.mean_[idx]) ** 2) / self.var_[idx],
                axis=1,
            )
            log_probs.append(log_prior + log_likelihood)
        scores = np.vstack(log_probs).T
        scores -= np.max(scores, axis=1, keepdims=True)
        probs = np.exp(scores)
        return probs / np.sum(probs, axis=1, keepdims=True)

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.classes_ is None:
            raise RuntimeError("GaussianNB must be fit before predict")
        return self.classes_[np.argmax(self.predict_proba(x), axis=1)]


class KNNClassifier:
    def __init__(self, k: int = 7, n_classes: int | None = None) -> None:
        self.k = k
        self.n_classes = n_classes
        self.x_: np.ndarray | None = None
        self.y_: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "KNNClassifier":
        self.x_ = x.copy()
        self.y_ = y.copy()
        if self.n_classes is None:
            self.n_classes = int(np.max(y)) + 1
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if self.x_ is None or self.y_ is None or self.n_classes is None:
            raise RuntimeError("KNNClassifier must be fit before predict")
        probs = np.zeros((x.shape[0], self.n_classes), dtype=float)
        k = min(self.k, self.x_.shape[0])
        for i, row in enumerate(x):
            dist = np.sum((self.x_ - row) ** 2, axis=1)
            nn = np.argpartition(dist, k - 1)[:k]
            weights = 1.0 / np.maximum(np.sqrt(dist[nn]), 1e-6)
            for cls, weight in zip(self.y_[nn], weights):
                probs[i, int(cls)] += weight
            probs[i] /= max(np.sum(probs[i]), EPS)
        return probs

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(x), axis=1)


@dataclass
class _TreeNode:
    feature: int | None = None
    threshold: float | None = None
    left: "_TreeNode | None" = None
    right: "_TreeNode | None" = None
    value: np.ndarray | float | None = None


class DecisionTreeClassifier:
    def __init__(
        self,
        max_depth: int = 8,
        min_samples_split: int = 8,
        min_samples_leaf: int = 3,
        max_features: int | None = None,
        n_thresholds: int = 16,
        random_state: int = 0,
        n_classes: int | None = None,
    ) -> None:
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.n_thresholds = n_thresholds
        self.random_state = random_state
        self.n_classes = n_classes
        self.root_: _TreeNode | None = None
        self.feature_importances_: np.ndarray | None = None
        self.rng_ = np.random.default_rng(random_state)

    def fit(self, x: np.ndarray, y: np.ndarray) -> "DecisionTreeClassifier":
        if self.n_classes is None:
            self.n_classes = int(np.max(y)) + 1
        self.feature_importances_ = np.zeros(x.shape[1], dtype=float)
        self.root_ = self._build(x, y, depth=0)
        total = float(np.sum(self.feature_importances_))
        if total > 0:
            self.feature_importances_ /= total
        return self

    def _leaf_value(self, y: np.ndarray) -> np.ndarray:
        counts = np.bincount(y, minlength=self.n_classes).astype(float)
        return (counts + 1e-3) / (float(np.sum(counts)) + 1e-3 * self.n_classes)

    def _gini(self, y: np.ndarray) -> float:
        counts = np.bincount(y, minlength=self.n_classes).astype(float)
        p = counts / max(float(np.sum(counts)), EPS)
        return float(1.0 - np.sum(p**2))

    def _candidate_thresholds(self, values: np.ndarray) -> np.ndarray:
        unique = np.unique(values)
        if unique.size <= 1:
            return np.array([])
        if unique.size <= self.n_thresholds:
            return (unique[:-1] + unique[1:]) / 2.0
        quantiles = np.linspace(0.05, 0.95, self.n_thresholds)
        return np.unique(np.quantile(values, quantiles))

    def _best_split(self, x: np.ndarray, y: np.ndarray) -> tuple[int | None, float | None, float]:
        parent_impurity = self._gini(y)
        best_gain = 0.0
        best_feature = None
        best_threshold = None
        n_features = x.shape[1]
        max_features = self.max_features or n_features
        candidates = self.rng_.choice(n_features, size=min(max_features, n_features), replace=False)

        for feature in candidates:
            values = x[:, feature]
            for threshold in self._candidate_thresholds(values):
                left_mask = values <= threshold
                left_count = int(np.sum(left_mask))
                right_count = y.size - left_count
                if left_count < self.min_samples_leaf or right_count < self.min_samples_leaf:
                    continue
                left_impurity = self._gini(y[left_mask])
                right_impurity = self._gini(y[~left_mask])
                weighted = (left_count * left_impurity + right_count * right_impurity) / y.size
                gain = parent_impurity - weighted
                if gain > best_gain:
                    best_gain = gain
                    best_feature = int(feature)
                    best_threshold = float(threshold)
        return best_feature, best_threshold, best_gain

    def _build(self, x: np.ndarray, y: np.ndarray, depth: int) -> _TreeNode:
        node = _TreeNode(value=self._leaf_value(y))
        if (
            depth >= self.max_depth
            or y.size < self.min_samples_split
            or np.unique(y).size == 1
        ):
            return node

        feature, threshold, gain = self._best_split(x, y)
        if feature is None or threshold is None or gain <= 1e-10:
            return node

        mask = x[:, feature] <= threshold
        node.feature = feature
        node.threshold = threshold
        node.left = self._build(x[mask], y[mask], depth + 1)
        node.right = self._build(x[~mask], y[~mask], depth + 1)
        if self.feature_importances_ is not None:
            self.feature_importances_[feature] += gain * y.size
        return node

    def _predict_one(self, row: np.ndarray) -> np.ndarray:
        if self.root_ is None:
            raise RuntimeError("DecisionTreeClassifier must be fit before predict")
        node = self.root_
        while node.feature is not None and node.threshold is not None and node.left is not None and node.right is not None:
            node = node.left if row[node.feature] <= node.threshold else node.right
        return np.asarray(node.value, dtype=float)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return np.vstack([self._predict_one(row) for row in x])

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(x), axis=1)


class DecisionTreeRegressor:
    def __init__(
        self,
        max_depth: int = 8,
        min_samples_split: int = 8,
        min_samples_leaf: int = 3,
        max_features: int | None = None,
        n_thresholds: int = 16,
        random_state: int = 0,
    ) -> None:
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.n_thresholds = n_thresholds
        self.random_state = random_state
        self.root_: _TreeNode | None = None
        self.feature_importances_: np.ndarray | None = None
        self.rng_ = np.random.default_rng(random_state)

    def fit(self, x: np.ndarray, y: np.ndarray) -> "DecisionTreeRegressor":
        self.feature_importances_ = np.zeros(x.shape[1], dtype=float)
        self.root_ = self._build(x, y, depth=0)
        total = float(np.sum(self.feature_importances_))
        if total > 0:
            self.feature_importances_ /= total
        return self

    def _candidate_thresholds(self, values: np.ndarray) -> np.ndarray:
        unique = np.unique(values)
        if unique.size <= 1:
            return np.array([])
        if unique.size <= self.n_thresholds:
            return (unique[:-1] + unique[1:]) / 2.0
        return np.unique(np.quantile(values, np.linspace(0.05, 0.95, self.n_thresholds)))

    def _variance(self, y: np.ndarray) -> float:
        return float(np.var(y)) if y.size else 0.0

    def _best_split(self, x: np.ndarray, y: np.ndarray) -> tuple[int | None, float | None, float]:
        parent_var = self._variance(y)
        best_gain = 0.0
        best_feature = None
        best_threshold = None
        n_features = x.shape[1]
        max_features = self.max_features or n_features
        candidates = self.rng_.choice(n_features, size=min(max_features, n_features), replace=False)

        for feature in candidates:
            values = x[:, feature]
            for threshold in self._candidate_thresholds(values):
                left_mask = values <= threshold
                left_count = int(np.sum(left_mask))
                right_count = y.size - left_count
                if left_count < self.min_samples_leaf or right_count < self.min_samples_leaf:
                    continue
                weighted = (
                    left_count * self._variance(y[left_mask])
                    + right_count * self._variance(y[~left_mask])
                ) / y.size
                gain = parent_var - weighted
                if gain > best_gain:
                    best_gain = gain
                    best_feature = int(feature)
                    best_threshold = float(threshold)
        return best_feature, best_threshold, best_gain

    def _build(self, x: np.ndarray, y: np.ndarray, depth: int) -> _TreeNode:
        node = _TreeNode(value=float(np.mean(y)))
        if depth >= self.max_depth or y.size < self.min_samples_split or self._variance(y) <= 1e-8:
            return node
        feature, threshold, gain = self._best_split(x, y)
        if feature is None or threshold is None or gain <= 1e-10:
            return node
        mask = x[:, feature] <= threshold
        node.feature = feature
        node.threshold = threshold
        node.left = self._build(x[mask], y[mask], depth + 1)
        node.right = self._build(x[~mask], y[~mask], depth + 1)
        if self.feature_importances_ is not None:
            self.feature_importances_[feature] += gain * y.size
        return node

    def _predict_one(self, row: np.ndarray) -> float:
        if self.root_ is None:
            raise RuntimeError("DecisionTreeRegressor must be fit before predict")
        node = self.root_
        while node.feature is not None and node.threshold is not None and node.left is not None and node.right is not None:
            node = node.left if row[node.feature] <= node.threshold else node.right
        return float(node.value)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.asarray([self._predict_one(row) for row in x], dtype=float)


class BaggedTreesClassifier:
    def __init__(
        self,
        n_estimators: int = 35,
        max_depth: int = 9,
        min_samples_leaf: int = 3,
        random_state: int = 0,
        n_classes: int | None = None,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.n_classes = n_classes
        self.trees_: list[DecisionTreeClassifier] = []
        self.feature_importances_: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "BaggedTreesClassifier":
        rng = np.random.default_rng(self.random_state)
        self.n_classes = self.n_classes or int(np.max(y)) + 1
        max_features = max(1, int(np.sqrt(x.shape[1])))
        importances = np.zeros(x.shape[1], dtype=float)
        self.trees_ = []
        for i in range(self.n_estimators):
            idx = rng.integers(0, x.shape[0], size=x.shape[0])
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                max_features=max_features,
                random_state=self.random_state + i + 17,
                n_classes=self.n_classes,
            )
            tree.fit(x[idx], y[idx])
            self.trees_.append(tree)
            if tree.feature_importances_ is not None:
                importances += tree.feature_importances_
        total = float(np.sum(importances))
        self.feature_importances_ = importances / total if total > 0 else importances
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if not self.trees_:
            raise RuntimeError("BaggedTreesClassifier must be fit before predict")
        probs = np.mean([tree.predict_proba(x) for tree in self.trees_], axis=0)
        return probs / np.sum(probs, axis=1, keepdims=True)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(x), axis=1)


class BaggedTreesRegressor:
    def __init__(
        self,
        n_estimators: int = 35,
        max_depth: int = 9,
        min_samples_leaf: int = 3,
        random_state: int = 0,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.trees_: list[DecisionTreeRegressor] = []
        self.feature_importances_: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "BaggedTreesRegressor":
        rng = np.random.default_rng(self.random_state)
        max_features = max(1, int(np.sqrt(x.shape[1])))
        importances = np.zeros(x.shape[1], dtype=float)
        self.trees_ = []
        for i in range(self.n_estimators):
            idx = rng.integers(0, x.shape[0], size=x.shape[0])
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                max_features=max_features,
                random_state=self.random_state + i + 29,
            )
            tree.fit(x[idx], y[idx])
            self.trees_.append(tree)
            if tree.feature_importances_ is not None:
                importances += tree.feature_importances_
        total = float(np.sum(importances))
        self.feature_importances_ = importances / total if total > 0 else importances
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if not self.trees_:
            raise RuntimeError("BaggedTreesRegressor must be fit before predict")
        return np.mean([tree.predict(x) for tree in self.trees_], axis=0)


class RidgeRegressor:
    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha
        self.coef_: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "RidgeRegressor":
        x_aug = np.c_[np.ones(x.shape[0]), x]
        penalty = np.eye(x_aug.shape[1]) * self.alpha
        penalty[0, 0] = 0.0
        self.coef_ = np.linalg.pinv(x_aug.T @ x_aug + penalty) @ x_aug.T @ y
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("RidgeRegressor must be fit before predict")
        return np.c_[np.ones(x.shape[0]), x] @ self.coef_


class KNNRegressor:
    def __init__(self, k: int = 7) -> None:
        self.k = k
        self.x_: np.ndarray | None = None
        self.y_: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "KNNRegressor":
        self.x_ = x.copy()
        self.y_ = y.copy()
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.x_ is None or self.y_ is None:
            raise RuntimeError("KNNRegressor must be fit before predict")
        preds = []
        k = min(self.k, self.x_.shape[0])
        for row in x:
            dist = np.sum((self.x_ - row) ** 2, axis=1)
            nn = np.argpartition(dist, k - 1)[:k]
            weights = 1.0 / np.maximum(np.sqrt(dist[nn]), 1e-6)
            preds.append(float(np.sum(self.y_[nn] * weights) / np.sum(weights)))
        return np.asarray(preds, dtype=float)


class WeightedClassifierEnsemble:
    def __init__(self, models: dict[str, object], weights: dict[str, float]) -> None:
        self.models = models
        self.weights = weights

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        total = None
        weight_sum = 0.0
        for name, model in self.models.items():
            weight = self.weights.get(name, 0.0)
            if weight <= 0:
                continue
            probs = model.predict_proba(x)
            total = probs * weight if total is None else total + probs * weight
            weight_sum += weight
        if total is None or weight_sum <= EPS:
            raise RuntimeError("Classifier ensemble has no active models")
        total = total / weight_sum
        return total / np.sum(total, axis=1, keepdims=True)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(x), axis=1)


class WeightedRegressorEnsemble:
    def __init__(self, models: dict[str, object], weights: dict[str, float]) -> None:
        self.models = models
        self.weights = weights

    def predict(self, x: np.ndarray) -> np.ndarray:
        total = None
        weight_sum = 0.0
        for name, model in self.models.items():
            weight = self.weights.get(name, 0.0)
            if weight <= 0:
                continue
            pred = np.asarray(model.predict(x), dtype=float)
            total = pred * weight if total is None else total + pred * weight
            weight_sum += weight
        if total is None or weight_sum <= EPS:
            raise RuntimeError("Regressor ensemble has no active models")
        return total / weight_sum


@dataclass
class FaultModelBundle:
    feature_names: list[str]
    label_encoder: LabelEncoder
    preprocessor: Preprocessor
    classifier_models: dict[str, object]
    classifier_weights: dict[str, float]
    regressor_models: dict[str, object]
    regressor_weights: dict[str, float]
    sample_rate_hz: float
    window_seconds: float
    bandpass_low_hz: float
    bandpass_high_hz: float
    rul_target_log: bool = True

    def classifier(self) -> WeightedClassifierEnsemble:
        return WeightedClassifierEnsemble(self.classifier_models, self.classifier_weights)

    def regressor(self) -> WeightedRegressorEnsemble:
        return WeightedRegressorEnsemble(self.regressor_models, self.regressor_weights)

    def transform_features(self, x: np.ndarray) -> np.ndarray:
        return self.preprocessor.transform(x)
