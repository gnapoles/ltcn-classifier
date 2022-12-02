import numpy as np
from numpy import linalg as la

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn import linear_model


class LTCN(BaseEstimator, ClassifierMixin):
    """
    Long-term Cognitive Network classifier

    """

    def __init__(self, T=20, phi=0.8, method="ridge", function="sigmoid", alpha=1.0E-4):

        """

        Parameters
        ----------
            T         :  {int}, default 20
                         Number of iterations to be performed
            phi       :  {float}, default 0.8
                         Amount of non-linearity during reasoning.
            method    :  {String}, default 'inverse'
                         Regression approach ('inverse', 'ridge')
            function  :  {String}, default 'sigmoid'
                         Activation function ('sigmoid', 'hyperbolic')
            alpha :      {float}, default 1.0E-4
                         Positive penalization for L2-regularization

        """

        self.T = T
        self.phi = phi
        self.method = method
        self.alpha = alpha
        self.function = function

        self.W1 = None
        self.W2 = None
        self.model = None

    def fit(self, X_train, Y_train):

        """ Fit the model to the (X_train, Y_train) data.

        Parameters
        ----------
            X_train : {array-like} of shape (n_samples, n_features)
                      The data to be used as the input for the model.
            Y_train : {array-like} of shape (n_samples, n_features)
                      The data to be used as the expected output.
        Returns
        ----------
            LTCN : Fully trained LTCN model ready for usage.

        """

        if self.W1 is None:
            self.W1 = self.build_matrix(X_train)

        Z = self.reasoning(X_train)

        if self.method == 'inverse':
            self.W2 = np.matmul(np.linalg.pinv(self.add_bias(Z)), Y_train)
        elif self.method == 'ridge':
            clf = linear_model.Ridge(alpha=self.alpha)
            self.model = clf.fit(Z, Y_train)

        return self

    def reasoning(self, A):

        """ Perform the recurrent reasoning process.

        Parameters
        ----------
            A : {array-like} of shape (n_samples, n_features)
                The input data used for activating the neurons.
        Returns
        ----------
            Z : {array-like} of shape (n_samples, n_features)
                Matrix with the neurons' last activation values.
        """

        A0 = A
        Z = A0

        for t in range(self.T):
            A = self.phi * self.transform(np.matmul(A, self.W1)) + (1 - self.phi) * A0
            Z = np.concatenate((Z, A), axis=1)

        return Z

    def transform(self, X):

        """ Apply tha activation function to the hidden state.

        Parameters
        ----------
            X : {array-like} of shape (n_samples, n_features)
                The raw hidden state of the network for transformation.
        Returns
        ----------
            H : {array-like} of shape (n_samples, n_features)
                The transformed hidden state of the network.
        """

        if self.function == "sigmoid":
            return 1 / (1 + np.exp(-X))
        else:
            return np.tanh(X)

    def predict(self, X):

        """ Predict the output for the given input.

        Parameters
        ----------
            X :       {array-like} of shape (n_samples, n_features)
                      The input data for prediction purposes.
        Returns
        ----------
            Y :       {array-like} of shape (n_samples, n_classes)
                      The prediction for the input data.
        """

        if self.method == "inverse":
            return np.dot(self.add_bias(self.reasoning(X)), self.W2)
        elif self.method == "ridge":
            return self.model.predict(self.reasoning(X))

    def build_matrix(self, X):

        """ Return the normalized Pearson product-moment correlation coefficients.

        Parameters
        ----------
            X : {array-like} of shape (n_samples, n_samples)
                The raw hidden state of the network for transformation.
        Returns
        ----------
            W : {array-like} of shape (n_features, n_features)
                The normalized correlation coefficient matrix.
        """

        n, m = X.shape
        T1 = np.sum(X, axis=0)
        T3 = np.sum(X ** 2, axis=0)

        W = np.random.random((m, m))
        for i in range(0, m):
            for j in range(0, m):
                d = n * T3[i] - T1[i] ** 2
                if d != 0:
                    W[i, j] = (n * np.sum(X[:, i] * X[:, j]) - T1[i] * T1[j]) / d

        return W / la.norm(W)

    def add_bias(self, X):

        """ Concatenate a bias vector to a given matrix.

        Parameters
        ----------
            X :       {array-like} of shape (n_samples, n_features)
                      The matrix to be expanded with the bias vector.
        Returns
        ----------
            Y :       {array-like} of shape (n_samples, n_features+1)
                      The enlarged matrix with the bias vector.

        """
        return np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
