from abc import ABC, abstractmethod
import numpy as np # type: ignore
import math

from sklearn.utils.validation import (
    check_X_y,
    check_is_fitted
)
from sklearn.linear_model._base import (
    BaseEstimator,
    LinearClassifierMixin,
    SparseCoefMixin,
)

from edu.but.cpl.basis_exchange_algorithm import BasisExchangeAlgorithm



class CPLBaseClassifier(ABC, LinearClassifierMixin, SparseCoefMixin, BaseEstimator):
    """
    Classifier core, common parts for different types of CPL classifiers.
    """

    def __init__(
        self,
        C="auto",
        fit_intercept=True
    ):
        self.__verbose = 0

        # control parameters
        self.C = C
        self.fit_intercept = fit_intercept
        self.allow_feature_vectors_exit = True
        self.allow_feature_vectors_enter = True
        self.fvs_mask = None


    def pretty_hyperplane(self, features_names=None, precision=4):
        """
        Return the separating hyperplane in a more readable form, with ordered coefficient values.
        """
        check_is_fitted(self)
        features_names = features_names or [f"x{f}" for f in range(self.dim)]
        hyp = sorted(
            [(features_names[f], round(c, precision)) for f, c in \
                enumerate(self.coef_) if c != 0.0],
            key=lambda x: abs(x[1]),
            reverse=True
        )
        hyp += [('theta', round(-self.intercept_, 4))]
        return hyp


    def fit(self, X, y, sample_weight=None, reduce_fs=True, verbose=0):
        """
        Build and train a classifier from the training set (X, y).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target values, class labels.
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted.
        reduce_fs : bool, default True
            Whether to reduce the feature space after optimisation by removing
            the features associated with the artificial base vectors from the feature space.
        verbose : int, default 0
            Refers to process logging, 0 means no logging, 1 means show informations during processing.

        Returns
        -------
        self : CPLBaseClassifier
        """
        self._verbose = verbose
        self.set_at_start_point(X, y, sample_weight)
        self._strategy_fit()
        # if reduce_fs:
        #     self.remove_artificial_features()
        self.__init_model()
        self._verbose = 0
        return self
    

    @abstractmethod
    def _strategy_fit(self):
        """
        A concrete optimisation method. An abstract method that should be implemented
        in classes inheriting from CPLBaseClassifier.
        """
        pass


    def predict_proba(self, X):
        """
        Predict class probabilities for X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
            The class probabilities of the input samples. The order of the
            classes corresponds to that in the attribute classes_.
        """
        predictions_float = self.decision_function(X)
        def sigmoid(x): return 1 / (1 + math.exp(-2*x))
        vfunc = np.vectorize(sigmoid)
        proba = np.zeros((X.shape[0], len(self.classes_)), dtype=np.float64)
        proba[:,1] = vfunc(predictions_float)
        proba[:,0] = 1.-proba[:,1]
        return proba


    @property
    def feature_importances_(self):
        """
        Return the feature importances.
        The importance of a feature is computed as the (normalized) weight in decision rule brought by that feature.

        Returns
        -------
        feature_importances_ : ndarray of shape (n_features,)
            Normalized weights in classifier decision rule.
        """
        check_is_fitted(self)
        importances = np.zeros(self.dim)
        nonzero_coefs_mask = self.coef_ != 0.0
        importances[nonzero_coefs_mask] = np.abs(self.coef_[nonzero_coefs_mask]) * self.fs.feature_weight[nonzero_coefs_mask]
        normalizer = np.sum(importances)
        if normalizer > 0.0:
            importances = importances / normalizer
        return importances


    def set_at_start_point(self, X, y, sample_weight=None) -> 'CPLBaseClassifier':
        """
        Initialise the classifier, prepare the data and computational structures,
        set the state of the optimisation procedure at the start point. 

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target values, class labels.
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted.

        Returns
        -------
        self : CPLBaseClassifier
            This classifier.
        """
        X, y = check_X_y(X, y)
        classes_ = np.unique(y)
        if len(classes_) > 2:
            raise ValueError(
                f"y must contain no more than 2 label values, but has such values {classes_}")
        if len(classes_) == 1:
            if classes_.dtype == bool:
                class0 = not classes_[0]
            elif classes_.dtype in [int, float, complex]:
                class0 = -classes_[0]
            else:
                class0 = "~" + str(classes_[0])
            classes_ = np.array([class0, classes_[0]])
        self.classes_ = classes_
        y = (y == self.classes_[1])
        self.__init_procedure(X, y, sample_weight)
        self._strategy_set_at_start_point()
        self.__init_model()
        return self
    

    @abstractmethod
    def _strategy_set_at_start_point(self):
        """
        Additional initialisation specific to a particular optimisation strategy.
        An abstract method that should be implemented in classes inheriting from CPLBaseClassifier.
        """
        pass


    def __init_procedure(self, X, y, sample_weight=None):
        self.num, self.dim = X.shape

        self.step = 0

        # feature weights
        # self.feature_weight = np.abs(X).max(axis=0)
        self.feature_weight = np.ones(self.dim, dtype=float)
        if self.fit_intercept:
            self.feature_weight = np.append(self.feature_weight, [0.0])

        # sample weights
        if sample_weight is not None:
            self.sample_weight = np.array(sample_weight)
        else:
            weight_Cp = 0.5 / sum(y)
            weight_Cm = 0.5 / sum(~y)
            self.sample_weight = np.array([weight_Cp if label else weight_Cm for label in y])
            
        # feature_vectors
        def aug_x(x, label):
            if self.fit_intercept:
                ax = np.append(x, [-1.])
            else:
                ax = np.array(x)
            if not label:
                ax *= -1.
            return ax
        self.feature_vectors = np.array([aug_x(x,label) for x,label in zip(X,y)])

        # parameter lambda
        if self.C == "auto":
            self.p_lambda = 0.0005 / self.num
        else:
            self.p_lambda = 1.0 / self.C / self.num
        # optimisation procedure
        self._bea = BasisExchangeAlgorithm(self.feature_vectors, self.sample_weight, self.feature_weight, self.p_lambda)


    def __init_model(self):
        """
        Set the parameters used in the prediction
        """
        hyperplane = self._bea.hyperplane
        self.coef_ = np.zeros(self.dim, dtype=float)
        self.coef_[hyperplane['features']] = hyperplane['coefs']
        self.intercept_ = -hyperplane['theta']
        self.final_objective_ = self._bea.criterion_function_value



class SekwemClassifier(CPLBaseClassifier):    

    def _strategy_set_at_start_point(self):
        theta_index = self.feature_vectors.shape[1]-1 if self.fit_intercept else None
        self._bea.reset_procedure(theta_index=theta_index)


    def _strategy_fit(self):
        # assumption: procedure set at zero
        self._bea.optimize(self._verbose)



# class GenetClassifier(CPLBaseClassifier):

#     def __repr__(self) -> str:
#         return f"GenetClassifier(C={self._C}, use_theta={self._use_theta})"
    

#     def _strategy_set_at_start_point(self):
#         self.fs.set_empty()
#         self.bases.init_empty()


#     def _strategy_fit(self):
#         # assumption: procedure set at zero

#         # correction_vector
#         self.cv = (self.fvs.vectors * np.array([self.fvs.sample_weight]).T).sum(axis=0)
#         if self._optimization_mode == OptimizationMode.OPT_HYP_MODE:
#             self.cv += -self.p_lambda * self.uvs.ev * self.fs.feature_weight
        
#         # determination of the first feature
#         m = abs(self.cv).argmax()

#         # optimisation
#         while True:
#             if not self.__add_feature_and_prepare_to_optimisation(m):
#                 break
#             self._optimise()
#             m = self.__find_feature_to_extend()
#             if m is None:
#                 break


#     def __add_feature_and_prepare_to_optimisation(self, m) -> bool:
#         """
#         Add feature to feature space and prepare to optimisation.

#         Parameters
#         ----------
#         m : int
#             Index of added feature.
        
#         Returns
#         -------
#         result : bool
#             Whether the procedure can be optimised after the addition of the feature.
#         """
#         self.add_feature(m)
#         self.l = m
#         self.cv[m] = sum(self.fvs.vectors[self.fvs.on_positive_side,m] * self.fvs.sample_weight[self.fvs.on_positive_side])
#         if self._optimization_mode == OptimizationMode.OPT_HYP_MODE:
#             self.cv[m] += -self.p_lambda * self.uvs.ev[m] * self.fs.feature_weight[m]
#         self.products_B1_cv[m] = self.bases.dot_B1_realv(m, self.cv)
#         return self._check_stop_and_init_s()


#     def __find_feature_to_extend(self) -> int:
#         """
#         Find a feature to extend the feature space.
        
#         Returns
#         -------
#         m : int
#             Index of the feature or None.
#         """
#         fos = self.fs.features_outside_space
#         wc = (self.fvs.vectors[self.fvs.on_positive_side][:,fos] * np.array([self.fvs.sample_weight[self.fvs.on_positive_side]]).T).sum(axis=0)

#         ffvb = self.fs.features[self.bases.B_type[self.fs.features]]
#         wc -= (self.products_B1_cv[ffvb] * (self.fvs.vectors[self.bases.B_index[ffvb]][:,fos]).T).sum(axis=1)

#         if self._optimization_mode == OptimizationMode.OPT_HYP_MODE:
#             wc -= self.p_lambda * self.uvs.ev[fos] * self.fs.feature_weight[fos]
        
#         correction_mask = (wc < 0)
#         wc[correction_mask] = -wc[correction_mask] - 2 * self.p_lambda * self.fs.feature_weight[fos[correction_mask]]

#         if (len(wc) == 0) or (np.max(wc) < ZERO):
#             return None
#         else:
#             return fos[wc.argmax()]
