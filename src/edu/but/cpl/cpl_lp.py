import numpy as np
from scipy.optimize import linprog

class CPL_LP:
    """
    L1-regularized linear SVM (soft-margin) jako programowanie liniowe (LP):
        min  C * sum_i sample_weight_i * xi_i + sum_j u_j
        s.t. y_i (w^T x_i + b) >= 1 - xi_i
             xi_i >= 0
             -u_j <= w_j <= u_j,  u_j >= 0
    Zmienna optymalizacyjna: z = [w (n), b (1), xi (m), u (n)]
    """

    def __init__(self, C=1.0, class_weight_mode: str = "balanced"):
        self.C = float(C)
        if self.C <= 0:
            raise ValueError("C must be positive.")
        self.class_weight_mode = class_weight_mode

    def _encode_labels(self, y):
        classes = np.unique(y)
        if set(classes) == {0, 1}:
            y_bin = np.where(y == 1, 1.0, -1.0)
            out_classes = np.array([0, 1])  # [neg, pos]
        elif set(classes) == {-1, 1} or set(classes) == {-1.0, 1.0}:
            y_bin = y.astype(float)
            out_classes = np.array([-1, 1])  # [neg, pos]
        else:
            raise ValueError("y must contain binary labels in {0,1} or {-1,1}.")
        return y_bin, out_classes

    def fit(self, X, y, sample_weight=None):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        m, n = X.shape

        y_bin, classes = self._encode_labels(y)
        if sample_weight is None:
            if self.class_weight_mode not in {"balanced", "uniform"}:
                raise ValueError(
                    "class_weight_mode must be one of {'balanced', 'uniform'}."
                )
            if self.class_weight_mode == "uniform":
                sample_weight = np.full(m, 1.0 / m, dtype=float)
            else:
                # Match CPL default balancing: each class contributes ~0.5 total weight.
                pos = y_bin == 1.0
                neg = ~pos
                n_pos = int(np.sum(pos))
                n_neg = int(np.sum(neg))
                if n_pos == 0 or n_neg == 0:
                    sample_weight = np.full(m, 1.0 / m, dtype=float)
                else:
                    w_pos = 0.5 / n_pos
                    w_neg = 0.5 / n_neg
                    sample_weight = np.where(pos, w_pos, w_neg).astype(float)
        else:
            sample_weight = np.asarray(sample_weight, dtype=float).reshape(-1)
            if sample_weight.shape[0] != m:
                raise ValueError(
                    f"sample_weight must have shape ({m},), got {sample_weight.shape}"
                )
            if np.any(sample_weight < 0):
                raise ValueError("sample_weight must be non-negative.")
        self.sample_weight_ = sample_weight

        # Indeksy w wektorze zmiennych z = [w | b | xi | u]
        idx_w = slice(0, n)
        idx_b = n
        idx_xi = slice(n + 1, n + 1 + m)
        idx_u = slice(n + 1 + m, n + 1 + m + n)
        N = n + 1 + m + n  # 2n + m + 1

        # Funkcja celu: C * sum_i sample_weight_i * xi_i + sum(u)
        c = np.zeros(N)
        c[idx_xi] = self.C * sample_weight
        c[idx_u] = 1.0

        A_ub = []
        b_ub = []

        # Ograniczenia marginesu: -y_i (w^T x_i + b) - xi_i <= -1
        for i in range(m):
            row = np.zeros(N)
            row[idx_w] = -y_bin[i] * X[i]
            row[idx_b] = -y_bin[i]
            row[n + 1 + i] = -1.0  # xi_i
            A_ub.append(row)
            b_ub.append(-1.0)

        # Ograniczenia |w_j| <= u_j  <=>  w_j - u_j <= 0  oraz  -w_j - u_j <= 0
        for j in range(n):
            # w_j - u_j <= 0
            row1 = np.zeros(N)
            row1[j] = 1.0                 # w_j
            row1[n + 1 + m + j] = -1.0    # -u_j
            A_ub.append(row1)
            b_ub.append(0.0)

            # -w_j - u_j <= 0
            row2 = np.zeros(N)
            row2[j] = -1.0                # -w_j
            row2[n + 1 + m + j] = -1.0    # -u_j
            A_ub.append(row2)
            b_ub.append(0.0)

        A_ub = np.vstack(A_ub)
        b_ub = np.asarray(b_ub)

        # Przedziały zmiennych: w_j, b nieogran.; xi_i >= 0; u_j >= 0
        bounds = []
        bounds += [(-np.inf, np.inf)] * n     # w
        bounds += [(-np.inf, np.inf)]         # b
        bounds += [(0.0, np.inf)] * m         # xi
        bounds += [(0.0, np.inf)] * n         # u

        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")

        if not res.success:
            raise RuntimeError(f"linprog failed: {res.message}")

        z = res.x
        self.w_ = z[idx_w]
        self.b_ = z[idx_b]
        self.xi_ = z[idx_xi]
        self.u_ = z[idx_u]
        self.classes_ = classes
        self.loss_ = float(np.sum(self.sample_weight_ * self.xi_))
        self.l1_norm_raw_ = float(np.sum(self.u_))
        self.l1_norm_ = self.l1_norm_raw_
        self.final_objective_ = (self.loss_, self.l1_norm_)
        self.F = res.fun
        return self

    def decision_function(self, X):
        X = np.asarray(X, dtype=float)
        return X @ self.w_ + self.b_

    def predict(self, X):
        s = self.decision_function(X)
        # Przypisz klasy oryginalne: ujemne -> classes_[0], dodatnie -> classes_[1]
        return np.where(s >= 0.0, self.classes_[1], self.classes_[0])

# --- przykład użycia ---
# X, y = ...  # X: (m,n), y: w {0,1} albo {-1,1}
# clf = CPL_LP(C=1.0).fit(X, y)
# y_pred = clf.predict(X)
# scores = clf.decision_function(X)
# w, b = clf.w_, clf.b_
