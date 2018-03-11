from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin, ClassifierMixin

ThundersvmBase = BaseEstimator
ThundersvmRegressorBase = RegressorMixin
ThundersvmClassifierBase = ClassifierMixin

import numpy as np
import scipy.sparse as sp

from sklearn.utils import check_X_y, column_or_1d, check_array
from sklearn.utils.validation import _num_samples

from ctypes import *
from os import path


dirname = path.dirname(path.abspath(__file__))
thundersvm = CDLL(path.join(dirname, '../build/lib/libthundersvm.so'))

SVM_TYPE = ['c_svc', 'nu_svc', 'one_class', 'epsilon_svr', 'nu_svr']

class SvmModel(ThundersvmBase):
    def __init__(self, kernel, degree,
                 gamma, coef0, cost, nu, epsilon,
                 tol, probability, class_weight):
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.cost = cost
        self.nu = nu
        self.epsilon = epsilon
        self.tol = tol
        self.probability = probability
        self.class_weight = class_weight

    def label_validate(self, y):

        return column_or_1d(y, warn=True).astype(np.float64)

    def fit(self, X, y):
        sparse = sp.isspmatrix(X)
        self._sparse = sparse and not callable(self.kernel)
        X, y = check_X_y(X, y, dtype=np.float64, order='C', accept_sparse='csr')
        y = self.label_validate(y)

        solver_type = SVM_TYPE.index(self._impl)


        if self.gamma == 'auto':
            self._gamma = 1.0 / X.shape[1]
        else:
            self._gamma = self.gamma

        kernel = self.kernel

        fit = self._sparse_fit if self._sparse else self._dense_fit

        fit(X, y, solver_type, kernel)

        self.shape_fit_ = X.shape

        return self

    def _dense_fit(self, X, y, solver_type, kernel):

        X = np.asarray(X, dtype=np.float64, order='C')
        samples = X.shape[0]
        features = X.shape[1]
        X_1d = X.ravel()
        data = (c_float * X_1d.size)()
        data[:] = X_1d
        kernel_type = kernel
        label = (c_float * y.size)()
        label[:] = y

        if self.class_weight is None:
            weight_size = 0
            self.class_weight = dict()
        else:
            weight_size = len(self.class_weight)
        weight_label = (c_int * weight_size)()
        weight_label[:] = self.class_weight.keys()
        weight = (c_float * weight_size)()
        weight[:] = self.class_weight.values()

        self.model = thundersvm.dense_model_scikit(
            samples, features, data, label, solver_type,
            kernel_type, self.degree, c_float(self._gamma), c_float(self.coef0),
            c_float(self.cost), c_float(self.nu), c_float(self.tol),
            self.probability, weight_size, weight_label, weight)



    def _sparse_fit(self, X, y, solver_type, kernel):
        X.data = np.asarray(X.data, dtype=np.float64, order='C')
        X.sort_indices()

        kernel_type = kernel


        data = (c_float * X.data.size)()
        data[:] = X.data
        indices = (c_int * X.indices.size)()
        indices[:] = X.indices
        indptr = (c_int * X.indptr.size)()
        indptr[:] = X.indptr
        label = (c_float * y.size)()
        label[:] = y

        if self.class_weight is None:
            weight_size = 0
            self.class_weight = dict()
        else:
            weight_size = len(self.class_weight)
        weight_label = (c_int * weight_size)()
        weight_label[:] = self.class_weight.keys()
        weight = (c_float * weight_size)()
        weight[:] = self.class_weight.values()

        self.model = thundersvm.sparse_model_scikit(
                X.shape[0], data, indptr, indices, label, solver_type,
                kernel_type, self.degree, c_float(self._gamma), c_float(self.coef0),
                c_float(self.cost), c_float(self.nu), c_float(self.tol),
                self.probability, weight_size, weight_label, weight)



    def _validate_for_predict(self, X):
        # check_is_fitted(self, 'support_')

        X = check_array(X, accept_sparse='csr', dtype=np.float64, order="C")
        if self._sparse and not sp.isspmatrix(X):
            X = sp.csr_matrix(X)
        if self._sparse:
            X.sort_indices()

        if sp.issparse(X) and not self._sparse and not callable(self.kernel):
            raise ValueError(
                "cannot use sparse input in %r trained on dense data"
                % type(self).__name__)

        return X

    def predict(self, X):

        X = self._validate_for_predict(X)
        predict = self._sparse_predict if self._sparse else self._dense_predict
        return predict(X)

    def _dense_predict(self, X):

        self.predict_label_ptr = (c_float * X.shape[0])()
        X = np.asarray(X, dtype=np.float64, order='C')
        samples = X.shape[0]
        features = X.shape[1]
        X_1d = X.ravel()

        data = (c_float * X_1d.size)()
        data[:] = X_1d
        print(X.shape[0])
        thundersvm.dense_predict(
            samples, features, data,
            self.model,
            self.predict_label_ptr)
        predict_label = []
        print(X.shape[0])
        print(self.predict_label_ptr[0])
        for index in range(0, X.shape[0]):
            predict_label += [self.predict_label_ptr[index]]
        self.predict_label = np.asarray(predict_label)
        return self.predict_label

    def _sparse_predict(self, X):
        self.predict_label_ptr = (c_float * X.shape[0])()
        data = (c_float * X.data.size)()
        data[:] = X.data
        indices = (c_int * X.indices.size)()
        indices[:] = X.indices
        indptr = (c_int * X.indptr.size)()
        indptr[:] = X.indptr
        thundersvm.sparse_predict(
            X.shape[0], data, indptr, indices,
            self.model,
            self.predict_label_ptr)
        predict_label = []
        for index in range(0, X.shape[0]):
            predict_label += [self.predict_label_ptr[index]]
        self.predict_label = np.asarray(predict_label)
        return self.predict_label




class SVC(SvmModel, ClassifierMixin):
     _impl = 'c_svc'
     def __init__(self, kernel = 2, degree = 3,
                  gamma = 'auto', coef0 = 0.0, cost = 1.0,
                  tol = 0.001, probability = False, class_weight = None):
         super(SVC, self).__init__(
             kernel=kernel, degree=degree, gamma=gamma,
             coef0=coef0, cost=cost, nu=0., epsilon=0.,
             tol=tol, probability=probability,
             class_weight=class_weight)



class NuSVC(SvmModel, ClassifierMixin):
    _impl = 'nu_svc'
    def __init__(self, kernel = 2, degree = 3, gamma = 'auto',
                 coef0 = 0.0, nu = 0.5, tol = 0.001,
                 probability = False):
        super(NuSVC, self).__init__(
            kernel = kernel, degree = degree, gamma = gamma,
            coef0 = coef0, cost = 0., nu = nu, epsilon= 0.,
            tol = tol, probability = probability, class_weight = None
        )

class OneClassSVM(SvmModel):
    _impl = 'one_class'
    def __init__(self, kernel = 2, degree = 3, gamma = 'auto',
                 coef0 = 0.0, nu = 0.5, tol = 0.001):
        super(OneClassSVM, self).__init__(
            kernel = kernel, degree = degree, gamma = gamma,
            coef0 = coef0, cost = 0., nu = nu, epsilon = 0.,
            tol = tol, probability= False, class_weight = None
        )

    def fit(self, X, y=None):
        super(OneClassSVM, self).fit(X, np.ones(_num_samples(X)))

class SVR(SvmModel, RegressorMixin):
    _impl = 'epsilon_svr'
    def __init__(self, kernel = 2, degree = 3, gamma = 'auto',
                 coef0 = 0.0, cost = 1.0, epsilon = 0.1,
                 tol = 0.001, probability = False):
        super(SVR, self).__init__(
            kernel = kernel, degree = degree, gamma = gamma,
            coef0 = coef0, cost = cost, epsilon = epsilon,
            tol = tol, probability = probability, class_weight = None
        )

class NuSVR(SvmModel, RegressorMixin):
    _impl = 'nu_svr'
    def __init(self, kernel = 2, degree = 3, gamma = 'auto',
               coef0 = 0.0, cost = 1.0, tol = 0.001, probability = False):
        super(NuSVR, self).__init(
            kernel = kernel, degree = degree, gamma = gamma,
            coef0 = coef0, cost = cost, epsilon = 0.,
            tol = tol, probability = probability, class_weight = None
        )