import time
import warnings
from typing import Optional, List
import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import hinge_loss, log_loss
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_X_y, check_array
from sklearn.exceptions import NotFittedError
from sklearn.exceptions import ConvergenceWarning

from edu.but.cpl.cpl_lp import CPL_LP


class _FeatureSubsetEstimator:
    """Expose a refitted subset model as an estimator on the original input columns."""

    def __init__(self, estimator, selected_indices, input_dim):
        self.estimator = estimator
        self.selected_indices = np.asarray(selected_indices, dtype=int)
        subset_coef = np.asarray(estimator.coef_)
        if subset_coef.ndim == 1:
            subset_coef = subset_coef.reshape(1, -1)
        self.coef_ = np.zeros((subset_coef.shape[0], int(input_dim)), dtype=float)
        self.coef_[:, self.selected_indices] = subset_coef
        self.intercept_ = np.asarray(estimator.intercept_)
        self.classes_ = getattr(estimator, "classes_", None)

    def __getattr__(self, name):
        return getattr(self.estimator, name)

    def _subset(self, X):
        return X[:, self.selected_indices]

    def decision_function(self, X):
        return self.estimator.decision_function(self._subset(X))

    def predict(self, X):
        return self.estimator.predict(self._subset(X))

    def predict_proba(self, X):
        return self.estimator.predict_proba(self._subset(X))


class _CPLLPWrapper:
    def __init__(self, C: float, class_weight_mode: str = "balanced"):
        self.C = C
        self.class_weight_mode = class_weight_mode
        self.model_ = None

    def fit(self, X, y, sample_weight=None):
        if hasattr(X, "toarray"):
            X = X.toarray()
        y_enc = np.asarray(y)
        self.model_ = CPL_LP(C=self.C, class_weight_mode=self.class_weight_mode)
        self.model_.fit(X, y_enc, sample_weight=sample_weight)
        self.coef_ = np.asarray(self.model_.w_).reshape(1, -1)
        self.intercept_ = np.asarray([self.model_.b_])
        # Use common objective semantics and shape: (sum_i xi_i, ||w||_1).
        scores = np.asarray(self.model_.decision_function(X)).reshape(-1)
        pos_label = self.model_.classes_[1]
        y_pm = np.where(y_enc == pos_label, 1.0, -1.0)
        xi_sum = float(np.maximum(0.0, 1.0 - y_pm * scores).sum())
        l1_norm = float(np.abs(self.model_.w_).sum())
        self.final_objective_ = (xi_sum, l1_norm)
        return self

    def decision_function(self, X):
        if hasattr(X, "toarray"):
            X = X.toarray()
        return self.model_.decision_function(X)

    def predict(self, X):
        if hasattr(X, "toarray"):
            X = X.toarray()
        return self.model_.predict(X).astype(int)

class ComplexL1NeuronsClassifier(BaseEstimator, ClassifierMixin):
    """
    Klasyfikator 'complex layer':
    - kolejne neurony liniowe (L1-logreg albo L1-SVM albo CPL),
    - każdy neuron trenowany na aktualnym zbiorze cech,
    - po treningu neuronu usuwamy z puli cechy z niezerowymi wagami,
    - na końcu: głosowanie większościowe neuronów (vote)

    Parametry
    ---------
    n_neurons : int
        Maksymalna liczba neuronów (L).
    base_model : {"logreg", "svm", "cpl"}
        "logreg" -> LogisticRegression z L1,
        "svm"    -> LinearSVC z L1,
        "cpl"    -> convex piecewise-linear neuron solved by LP.
    C : float
        Parametr regularyzacji C dla neuronów.
    class_weight_mode : {"balanced", "uniform"}
        Tryb ważenia próbek dla wszystkich modeli bazowych.
    max_iter : int
        Maksymalna liczba iteracji.
    random_state : Optional[int]
        Ziarno RNG.
    zero_threshold : float
        Próg, poniżej którego współczynnik traktujemy jako 0.
    top_k_features : Optional[int]
        Jeżeli ustawione, wybierz do refitu najwyżej k niezerowych współczynników
        o największej wartości bezwzględnej.
    verbose : int
        0 – cisza; >0 – wypisuje info w trakcie fit.
    """

    def __init__(
        self,
        n_neurons: int = 5,
        base_model: str = "logreg",
        C: float = 1.0,
        class_weight_mode: str = "balanced",
        max_iter: int = 2000,
        random_state: Optional[int] = None,
        zero_threshold: float = 1e-8,
        top_k_features: Optional[int] = None,
        verbose: int = 0,
    ):
        self.n_neurons = n_neurons
        self.base_model = base_model
        self.C = C
        self.class_weight_mode = class_weight_mode
        self.max_iter = max_iter
        self.random_state = random_state
        self.zero_threshold = zero_threshold
        self.top_k_features = top_k_features
        self.verbose = verbose

    # ---------- konstrukcja bazowego neuronu ----------

    def _make_base_estimator(self, n_samples: Optional[int] = None):
        if self.base_model == "logreg":
            return LogisticRegression(
                penalty="l1",
                C=self.C,
                solver="saga",      # L1 + multiclass
                max_iter=self.max_iter,
                random_state=self.random_state,
            )
        elif self.base_model == "svm":
            return LinearSVC(
                penalty="l1",
                loss="squared_hinge",
                dual=False,
                C=self.C,
                max_iter=self.max_iter,
                random_state=self.random_state,
            )
        elif self.base_model == "cpl":
            return _CPLLPWrapper(C=self.C, class_weight_mode=self.class_weight_mode)
        else:
            raise ValueError(
                f"Unknown base_model='{self.base_model}' (use 'logreg', 'svm', or 'cpl')."
            )

    def _build_sample_weight(self, y_enc: np.ndarray) -> np.ndarray:
        mode = self.class_weight_mode
        n = int(y_enc.shape[0])
        if n == 0:
            raise ValueError("Empty target array.")
        if mode == "uniform":
            return np.full(n, 1.0 / n, dtype=float)
        if mode == "balanced":
            classes, counts = np.unique(y_enc, return_counts=True)
            k = len(classes)
            weights = np.zeros(n, dtype=float)
            for c, cnt in zip(classes, counts):
                if cnt <= 0:
                    continue
                weights[y_enc == c] = 1.0 / (k * cnt)
            return weights
        raise ValueError("class_weight_mode must be one of {'balanced', 'uniform'}.")

    def _compute_objective_tuple(self, base_clf, X_sub, y_enc, coef) -> Optional[tuple]:
        existing = getattr(base_clf, "final_objective_", None)
        if isinstance(existing, tuple) and len(existing) >= 2:
            return existing

        l1_norm = float(np.abs(coef).sum())

        if self.base_model == "logreg":
            proba = base_clf.predict_proba(X_sub)
            loss = float(log_loss(y_enc, proba, labels=[0, 1]))
            return (loss, l1_norm)

        if self.base_model == "svm":
            scores = np.asarray(base_clf.decision_function(X_sub)).reshape(-1)
            y_pm = np.where(y_enc == 1, 1, -1)
            loss = float(hinge_loss(y_pm, scores, labels=[-1, 1]))
            return (loss, l1_norm)

        return None

    def _compute_common_objective_tuple(self, base_clf, X_sub, y_enc, coef) -> Optional[tuple]:
        # Common scale across models: empirical loss on training subset + raw L1 norm of coefficients.
        l1_norm = float(np.abs(coef).sum())

        if self.base_model == "logreg":
            proba = base_clf.predict_proba(X_sub)
            loss = float(log_loss(y_enc, proba, labels=[0, 1]))
            return (loss, l1_norm)

        scores = np.asarray(base_clf.decision_function(X_sub)).reshape(-1)
        y_pm = np.where(y_enc == 1, 1, -1)
        loss = float(hinge_loss(y_pm, scores, labels=[-1, 1]))
        return (loss, l1_norm)

    # ---------- fit ----------

    @staticmethod
    def _fit_with_diagnostics(estimator, X, y, sample_weight):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            estimator.fit(X, y, sample_weight=sample_weight)
        converged = not any(
            issubclass(item.category, ConvergenceWarning) for item in caught
        )
        n_iter = getattr(estimator, "n_iter_", None)
        if n_iter is not None:
            values = np.asarray(n_iter).reshape(-1)
            n_iter = int(np.max(values)) if values.size else None
        return converged, n_iter

    def fit(self, X, y):
        X, y = check_X_y(X, y, accept_sparse="csr")
        if self.top_k_features is not None and self.top_k_features < 1:
            raise ValueError("top_k_features must be >= 1 or None.")
        self.n_features_in_ = X.shape[1]

        # kodowanie etykiet na 0..K-1
        self._label_encoder = LabelEncoder().fit(y)
        y_enc = self._label_encoder.transform(y)
        self.classes_ = self._label_encoder.classes_
        self.n_classes_ = len(self.classes_)

        if self.n_classes_ > 2:
            raise ValueError("ComplexL1NeuronsClassifier wspiera tylko klasyfikację binarną.")

        remaining_features = np.arange(self.n_features_in_, dtype=int)

        self.neurons_: List = []
        self.neuron_fit_times_sec_: List[float] = []
        self.neuron_fit_diagnostics_: List[dict] = []
        self._input_feature_groups_: List[np.ndarray] = []   # cechy, na których neuron był trenowany
        self.feature_groups_: List[np.ndarray] = []          # cechy z niezerowymi wagami (dla interpretacji)

        for i in range(self.n_neurons):
            if remaining_features.size == 0:
                if self.verbose:
                    print(f"[ComplexL1] Brak pozostałych cech, zatrzymuję się na neuronie {i}.")
                break

            if self.verbose:
                print(f"[ComplexL1] Trening neuronu {i+1}/{self.n_neurons} na "
                      f"{remaining_features.size} cechach.")

            # cechy wejściowe tego neuronu (to, co on realnie widzi)
            input_features = remaining_features.copy()
            base_clf = self._make_base_estimator(n_samples=X.shape[0])

            X_sub = X[:, input_features]
            sample_weight = self._build_sample_weight(y_enc)
            neuron_fit_start = time.perf_counter()
            selection_converged, selection_n_iter = self._fit_with_diagnostics(
                base_clf, X_sub, y_enc, sample_weight
            )
            refit_performed = False
            refit_converged = None
            refit_n_iter = None

            initial_coef = np.asarray(getattr(base_clf, "coef_", []))
            if initial_coef.ndim == 1:
                initial_coef = initial_coef.reshape(1, -1)
            initial_used = np.flatnonzero(
                np.any(np.abs(initial_coef) > self.zero_threshold, axis=0)
            )
            if (
                self.top_k_features is not None
                and initial_used.size > self.top_k_features
            ):
                importance = np.max(np.abs(initial_coef[:, initial_used]), axis=0)
                ranked = sorted(
                    zip(initial_used.tolist(), importance.tolist()),
                    key=lambda item: (-item[1], item[0]),
                )
                refit_indices = np.asarray(
                    [index for index, _ in ranked[: self.top_k_features]], dtype=int
                )
                refit_clf = self._make_base_estimator(n_samples=X.shape[0])
                refit_converged, refit_n_iter = self._fit_with_diagnostics(
                    refit_clf, X_sub[:, refit_indices], y_enc, sample_weight
                )
                refit_performed = True
                base_clf = _FeatureSubsetEstimator(
                    refit_clf, refit_indices, X_sub.shape[1]
                )
            neuron_fit_time_sec = float(time.perf_counter() - neuron_fit_start)

            coef = getattr(base_clf, "coef_", None)
            if coef is None:
                raise RuntimeError("Base estimator must expose coef_ after fitting.")
            coef = np.asarray(coef)
            if coef.ndim == 1:
                coef = coef.reshape(1, -1)

            objective = self._compute_objective_tuple(base_clf, X_sub, y_enc, coef)
            if objective is not None:
                base_clf.final_objective_ = objective
            common_objective = self._compute_common_objective_tuple(base_clf, X_sub, y_enc, coef)
            if common_objective is not None:
                base_clf.common_objective_ = common_objective

            coef_threshold = self.zero_threshold
            used_mask = np.any(np.abs(coef) > coef_threshold, axis=0)

            if not np.any(used_mask):
                if self.verbose:
                    print(f"[ComplexL1] Neuron {i+1} nie wybrał żadnej cechy, przerywam.")
                break

            selected_features = input_features[used_mask]

            if self.verbose:
                print(f"[ComplexL1] Neuron {i+1} wybrał {selected_features.size} cech.")

            self.neurons_.append(base_clf)
            self.neuron_fit_times_sec_.append(neuron_fit_time_sec)
            self.neuron_fit_diagnostics_.append(
                {
                    "selection_converged": selection_converged,
                    "selection_n_iter": selection_n_iter,
                    "refit_performed": refit_performed,
                    "refit_converged": refit_converged,
                    "refit_n_iter": refit_n_iter,
                }
            )
            self._input_feature_groups_.append(input_features)
            self.feature_groups_.append(selected_features)

            # usuwamy wybrane cechy z puli dla kolejnych neuronów
            remaining_features = remaining_features[~used_mask]

        self.n_neurons_ = len(self.neurons_)
        if self.n_neurons_ == 0:
            raise RuntimeError("Żaden neuron nie został zbudowany – spróbuj zmienić C "
                               "albo preprocessing danych.")

        # Ustawiamy flagę fitted przed obliczeniem wyjść do meta-klasyfikatora
        self.is_fitted_ = True

        return self

    # ---------- helpers ----------

    def _check_is_fitted(self):
        if not hasattr(self, "is_fitted_") or not self.is_fitted_:
            raise NotFittedError(
                "ComplexL1NeuronsClassifier nie został jeszcze wytrenowany. "
                "Najpierw wywołaj .fit(X, y)."
            )

    def _vote_counts(self, X):
        P = self._vote_matrix(X)
        counts = np.zeros((P.shape[0], self.n_classes_), dtype=float)
        for i in range(P.shape[0]):
            counts[i, :] = np.bincount(P[i, :], minlength=self.n_classes_)
        return counts

    def _vote_matrix(self, X):
        X = check_array(X, accept_sparse="csr")
        if not hasattr(self, "n_neurons_") or self.n_neurons_ == 0:
            raise RuntimeError("Brak wytrenowanych neuronów do głosowania.")

        all_preds = []
        for clf, input_idx in zip(self.neurons_, self._input_feature_groups_):
            X_sub = X[:, input_idx]
            y_j = clf.predict(X_sub)
            all_preds.append(y_j.reshape(-1, 1))

        return np.hstack(all_preds).astype(int)   # [n_samples, n_neurons_]

    # ---------- API sklearn ----------

    def predict(self, X):
        self._check_is_fitted()
        P = self._vote_matrix(X)
        return self._predict_from_votes(P, P.shape[1])

    def predict_proba(self, X):
        """
        Zwraca rozkład prawdopodobieństw znormalizowany z liczby głosów.
        (to nie jest „czyste” LR-proba, tylko prosta agregacja vote)
        """
        self._check_is_fitted()
        counts = self._vote_counts(X)
        return counts / counts.sum(axis=1, keepdims=True)

    def decision_function(self, X):
        raise AttributeError("decision_function nie jest dostępne dla głosowania (vote).")

    def vote_matrix(self, X):
        self._check_is_fitted()
        return self._vote_matrix(X)

    def score_matrix(self, X):
        """Return one raw linear decision score per sample and fitted neuron."""
        self._check_is_fitted()
        X = check_array(X, accept_sparse="csr")
        all_scores = []
        for clf, input_idx in zip(self.neurons_, self._input_feature_groups_):
            X_sub = X[:, input_idx]
            scores = np.asarray(clf.decision_function(X_sub), dtype=float).reshape(-1)
            all_scores.append(scores.reshape(-1, 1))
        return np.hstack(all_scores)

    def predict_with_votes(self, P, n_neurons: int):
        self._check_is_fitted()
        P = np.asarray(P)
        if P.ndim != 2:
            raise ValueError("P must be a 2D vote matrix.")
        return self._predict_from_votes(P, n_neurons)

    # ---------- gettery ----------

    def get_feature_groups(self) -> List[np.ndarray]:
        """
        Zwraca listę wektorów indeksów cech (w oryginalnym X),
        które dany neuron faktycznie wykorzystuje (niezerowe wagi).
        """
        self._check_is_fitted()
        return self.feature_groups_

    def get_input_feature_groups(self) -> List[np.ndarray]:
        """
        Zwraca listę wektorów indeksów cech, na których dany neuron był trenowany
        (czyli "pula dostępnych cech" na jego etapie).
        """
        self._check_is_fitted()
        return self._input_feature_groups_

    def get_neurons(self) -> List:
        """Zwraca listę wytrenowanych bazowych klasyfikatorów (neuronów)."""
        self._check_is_fitted()
        return self.neurons_

    def get_neuron_fit_times_sec(self) -> List[float]:
        """Return wall-clock fit time for each successfully fitted neuron."""
        self._check_is_fitted()
        return list(self.neuron_fit_times_sec_)

    def get_neuron_fit_diagnostics(self) -> List[dict]:
        """Return convergence diagnostics for selection and optional top-k refit."""
        self._check_is_fitted()
        return [dict(item) for item in self.neuron_fit_diagnostics_]

    def _predict_from_votes(self, P, n_neurons: int):
        if n_neurons < 1:
            raise ValueError("n_neurons must be >= 1.")
        n_effective = min(n_neurons, P.shape[1])
        Pn = P[:, :n_effective]
        counts = np.zeros((Pn.shape[0], self.n_classes_), dtype=float)
        for i in range(Pn.shape[0]):
            counts[i, :] = np.bincount(Pn[i, :], minlength=self.n_classes_)

        max_counts = counts.max(axis=1, keepdims=True)
        tie_mask = (counts == max_counts).sum(axis=1) > 1

        y_enc = np.argmax(counts, axis=1).astype(int)
        # Remis rozstrzygamy na korzyść pierwszego neuronu (największa pula cech).
        if np.any(tie_mask):
            y_enc[tie_mask] = Pn[tie_mask, 0]

        return self._label_encoder.inverse_transform(y_enc)










class ComplexL1OvRClassifier(BaseEstimator, ClassifierMixin):
    """
    One-vs-Rest wrapper wokół ComplexL1NeuronsClassifier.
    Dla K klas uczy K binarnych complex layers: (klasa k) vs (reszta).
    """

    def __init__(
        self,
        n_neurons: int = 5,
        base_model: str = "logreg",
        C: float = 1.0,
        class_weight_mode: str = "balanced",
        max_iter: int = 2000,
        random_state: Optional[int] = None,
        zero_threshold: float = 1e-8,
        top_k_features: Optional[int] = None,
        verbose: int = 0,
    ):
        self.n_neurons = n_neurons
        self.base_model = base_model
        self.C = C
        self.class_weight_mode = class_weight_mode
        self.max_iter = max_iter
        self.random_state = random_state
        self.zero_threshold = zero_threshold
        self.top_k_features = top_k_features
        self.verbose = verbose

    def _make_base_complex(self):
        # zakładamy, że ComplexL1NeuronsClassifier jest zdefiniowany
        return ComplexL1NeuronsClassifier(
            n_neurons=self.n_neurons,
            base_model=self.base_model,
            C=self.C,
            class_weight_mode=self.class_weight_mode,
            max_iter=self.max_iter,
            random_state=self.random_state,
            zero_threshold=self.zero_threshold,
            top_k_features=self.top_k_features,
            verbose=self.verbose,
        )

    def fit(self, X, y):
        X, y = check_X_y(X, y, accept_sparse="csr")
        self.n_features_in_ = X.shape[1]
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        self.estimators_: List[ComplexL1NeuronsClassifier] = []

        for c in self.classes_:
            if self.verbose:
                print(f"[OvR] Trening complex layer dla klasy {c} vs reszta")
            y_bin = (y == c).astype(int)
            est = self._make_base_complex()
            est.fit(X, y_bin)   # tu complex layer rozwiązuje problem binarny
            self.estimators_.append(est)

        self.is_fitted_ = True
        return self

    def _check_is_fitted(self):
        if not hasattr(self, "is_fitted_") or not self.is_fitted_:
            raise NotFittedError(
                "ComplexL1OvRClassifier nie został jeszcze wytrenowany. "
                "Najpierw wywołaj .fit(X, y)."
            )

    def decision_function(self, X):
        """
        Zwraca macierz [n_samples, n_classes_] ze score'ami dla każdej klasy (OvR).
        """
        self._check_is_fitted()
        X = check_array(X, accept_sparse="csr")

        scores = []
        for est in self.estimators_:
            if hasattr(est, "decision_function"):
                s = est.decision_function(X)
                s = np.asarray(s).reshape(-1)
            elif hasattr(est, "predict_proba"):
                proba = est.predict_proba(X)
                if proba.shape[1] == 2:
                    s = proba[:, 1]          # prawd. klasy „1” (positive)
                else:
                    s = np.max(proba, axis=1)
            else:
                s = est.predict(X)           # fallback: etykiety 0/1 jako score

            scores.append(s.reshape(-1, 1))

        return np.hstack(scores)             # [n_samples, n_classes_]

    def predict(self, X):
        scores = self.decision_function(X)
        idx = np.argmax(scores, axis=1)
        return self.classes_[idx]

    def predict_proba(self, X):
        """
        Zamienia score'y OvR na rozkład prawdopodobieństw przez softmax.
        (to nie jest „czyste” LR-proba, ale sensowne porównywalne skale).
        """
        scores = self.decision_function(X)
        scores = scores - scores.max(axis=1, keepdims=True)  # stabilizacja numeryczna
        exp_scores = np.exp(scores)
        proba = exp_scores / exp_scores.sum(axis=1, keepdims=True)
        return proba
