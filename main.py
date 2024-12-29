import numpy as np
import pandas as pd
from minisom import MiniSom
from sklearn.neural_network import MLPClassifier

class SOEDClassifier:
    def __init__(self, 
                 mlp_hidden_layer_sizes=(100,), mlp_activation='relu', mlp_solver='adam',
                 mlp_alpha=0.0001, mlp_batch_size='auto', mlp_learning_rate='constant',
                 mlp_max_iter=200, mlp_tol=1e-4, mlp_verbose=False,
                 mlp_warm_start=False, mlp_n_iter_no_change=10, mlp_beta_1=0.9, mlp_beta_2=0.999, mlp_epsilon=1e-8,
                 som_x=10, som_y=10, som_input_len=None, som_sigma=1.0, som_learning_rate=0.5,
                 som_decay_function=None, som_neighborhood_function='gaussian', som_n_iter = 100,
                 random_state =None):
        """
        Custom Multi-Layer Perceptron (MLP) Classifier with MiniSOM for feature transformation.

        Parameters:
        - MLPClassifier hyperparameters:
            - mlp_hidden_layer_sizes: tuple, default=(100,)
            - mlp_activation: {'identity', 'logistic', 'tanh', 'relu'}, default='relu'
            - mlp_solver: {'lbfgs', 'sgd', 'adam'}, default='adam'
            - mlp_alpha: float, default=0.0001
            - mlp_batch_size: int or 'auto', default='auto'
            - mlp_learning_rate: {'constant', 'invscaling', 'adaptive'}, default='constant'
            - mlp_max_iter: int, default=200
            - mlp_tol: float, default=1e-4
            - mlp_verbose: bool, default=False
            - mlp_warm_start: bool, default=False
            - mlp_n_iter_no_change: int, default=10
            - mlp_beta_1: float, default=0.9
            - mlp_beta_2: float, default=0.999
            - mlp_epsilon: float, default=1e-8

        - MiniSOM hyperparameters:
            - som_x: int, default=10
            - som_y: int, default=10
            - som_input_len: int or None, default=None
            - som_sigma: float, default=1.0
            - som_learning_rate: float, default=0.5
            - som_decay_function: callable or None, default=None
            - som_neighborhood_function: {'gaussian', 'mexican_hat'}, default='gaussian'
        """

        assert type(som_input_len)==int, 'som_input_len needs to be integer.'
        self.mlp_hidden_layer_sizes = mlp_hidden_layer_sizes
        self.mlp_activation = mlp_activation
        self.mlp_solver = mlp_solver
        self.mlp_alpha = mlp_alpha
        self.mlp_batch_size = mlp_batch_size
        self.mlp_learning_rate = mlp_learning_rate
        self.mlp_max_iter = mlp_max_iter
        self.mlp_random_state = random_state
        self.mlp_tol = mlp_tol
        self.mlp_verbose = mlp_verbose
        self.mlp_warm_start = mlp_warm_start
        self.mlp_n_iter_no_change = mlp_n_iter_no_change
        self.mlp_beta_1 = mlp_beta_1
        self.mlp_beta_2 = mlp_beta_2
        self.mlp_epsilon = mlp_epsilon

        self.som_x = som_x
        self.som_y = som_y
        self.som_input_len = som_input_len + 3
        self.som_sigma = som_sigma
        self.som_learning_rate = som_learning_rate
        self.som_decay_function = som_decay_function
        self.som_neighborhood_function = som_neighborhood_function
        self.som_random_seed = random_state
        self.som_n_iter = som_n_iter

        #if self.som_random_seed is None:
        #    self.som_random_seed = np.random.randint(1000000)

        self.mlp = MLPClassifier(hidden_layer_sizes=self.mlp_hidden_layer_sizes, activation=self.mlp_activation,
                                 solver=self.mlp_solver, alpha=self.mlp_alpha, batch_size=self.mlp_batch_size,
                                 learning_rate=self.mlp_learning_rate, max_iter=self.mlp_max_iter,
                                 random_state=self.mlp_random_state, tol=self.mlp_tol, verbose=self.mlp_verbose,
                                 warm_start=self.mlp_warm_start, n_iter_no_change=self.mlp_n_iter_no_change,
                                 beta_1=self.mlp_beta_1, beta_2=self.mlp_beta_2, epsilon=self.mlp_epsilon)

        self.som = MiniSom(self.som_x, self.som_y, self.som_input_len, sigma=self.som_sigma,
                           learning_rate=self.som_learning_rate,
                           neighborhood_function=self.som_neighborhood_function, random_seed=self.som_random_seed)

        self.is_fitted = False

    def fit(self, X, y, c=None):
        """
        Fit the model to the training data.

        Parameters:
        - X: array-like of shape (n_samples, n_features)
            The input data.
        - y: array-like of shape (n_samples,)
            The target values.
        - c: array_like of shale (n_samples, 2)
            The cost of mistake.

        Returns:
        - self: object
            Fitted estimator.
        """

        if c is None:
            c = np.ones([X.shape[0],2])

        assert X.shape[0] == y.shape[0], 'X and y must be of the same length.'
        assert X.shape[0] == c.shape[0], 'X and c must be of the same length.'
        assert c.shape[1] == 2, 'c must have two columns.'
        assert np.isin(y, [0, 1]).all(), 'y can only have binary values.'

        _continue = True
        _multiplier = 0.5
        while _continue:
            X_som = np.column_stack((X, y*_multiplier,c*_multiplier))

            # Train the MiniSom
            self.som.train(X_som, self.som_n_iter)
            winner_coordinates = np.array([self.som.winner(x) for x in X_som])
            df = pd.DataFrame(np.column_stack((winner_coordinates,y)),columns=['X','Y','L'])
            pure_df = df.groupby(['X','Y','L']).size().unstack()
            is_pure = all(pure_df.loc[r].nunique(dropna=True) <= 1 for r in pure_df.index)

            _continue = not is_pure
            _multiplier += 0.25

        # Transform features using MiniSom
        X_transformed = np.array([self.som.winner(x) for x in X])

        # Train the MLPClassifier
        self.mlp.fit(X_transformed, y)
        self.is_fitted = True
        print("Model training complete.")
        return self

    def predict(self, X):
        """
        Predict class labels for samples in X.

        Parameters:
        - X: array-like of shape (n_samples, n_features)
            The input data.

        Returns:
        - y_pred: array-like of shape (n_samples,)
            Predicted class labels.
        """
        if not self.is_fitted:
            raise ValueError("The model has not been fitted yet.")
        
        # Transform features using MiniSom
        X_transformed = np.array([self.som.winner(x) for x in X])

        # Predict using the trained MLPClassifier
        y_pred = self.mlp.predict(X_transformed)
        return y_pred

    def predict_proba(self, X):
        """
        Probability estimates for samples in X.

        Parameters:
        - X: array-like of shape (n_samples, n_features)
            The input data.

        Returns:
        - proba: array-like of shape (n_samples, n_classes)
            Probability estimates.
        """
        if not self.is_fitted:
            raise ValueError("The model has not been fitted yet.")

        # Transform features using MiniSom
        X_transformed = np.array([self.som.winner(x) for x in X])

        # Predict probabilities using the trained MLPClassifier
        proba = self.mlp.predict_proba(X_transformed)
        return proba

    def score(self, X, y):
        """
        Returns the mean accuracy on the given test data and labels.

        Parameters:
        - X: array-like of shape (n_samples, n_features)
            Test samples.
        - y: array-like of shape (n_samples,)
            True labels for X.

        Returns:
        - score: float
            Mean accuracy of self.predict(X) wrt. y.
        """
        y_pred = self.predict(X)
        return np.mean(y_pred == y)

# Example usage:
# clf = CustomMLPClassifier(mlp_hidden_layer_sizes=(50, 30), mlp_max_iter=500, som_x=5, som_y=5, som_input_len=4, som_random_seed=42)
# clf.fit(X_train, y_train)
# y_pred = clf.predict(X_test)
# y_proba = clf.predict_proba(X_test)
