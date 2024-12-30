import numpy as np
import pandas as pd
from minisom import MiniSom
from sklearn.neural_network import MLPRegressor

def fill_nan_with_direct_neighbors(matrix):
    # Ensure the input is a numeric type
    if not np.issubdtype(matrix.dtype, np.number):
        matrix = matrix.astype(float)  # Convert to float if needed

    # Create a padded version of the matrix to handle edges
    padded_matrix = np.pad(matrix, pad_width=1, mode='constant', constant_values=np.nan)
    rows, cols = matrix.shape

    # Initialize an output matrix
    filled_matrix = matrix.copy()

    # Define the weights for the 3x3 neighborhood (only direct neighbors have weight)
    weights = np.array([
        [0.0, 1.0, 0.0],  # Top row: Direct only (up)
        [1.0, 0.0, 1.0],  # Middle row: Direct (left, right)
        [0.0, 1.0, 0.0]   # Bottom row: Direct only (down)
    ])

    # Iterate over each cell in the original matrix
    for i in range(rows):
        for j in range(cols):
            if np.isnan(matrix[i, j]):  # Only process NaN values
                # Extract the 3x3 neighborhood
                neighborhood = padded_matrix[i:i+3, j:j+3]

                # Apply weights and mask valid neighbors
                weighted_values = neighborhood * weights
                valid_mask = ~np.isnan(weighted_values)

                # Calculate the weighted average of valid neighbors
                if np.any(valid_mask):
                    filled_matrix[i, j] = np.sum(weighted_values[valid_mask]) / np.sum(weights[valid_mask])

    return filled_matrix

def fill_nan_with_weighted_neighbors(matrix):
    # Ensure the input is a numeric type
    if not np.issubdtype(matrix.dtype, np.number):
        matrix = matrix.astype(float)  # Convert to float if necessary

    # Create a padded version of the matrix to handle edges
    padded_matrix = np.pad(matrix, pad_width=1, mode='constant', constant_values=np.nan)
    rows, cols = matrix.shape

    # Initialize an output matrix
    filled_matrix = matrix.copy()

    # Define the weights for the 3x3 neighborhood
    weights = np.array([
        [0.25, 1.00, 0.25],  # Top row: Direct neighbors (up)
        [1.00, 0.00, 1.00],  # Middle row: Direct neighbors (left, right)
        [0.25, 1.00, 0.25]   # Bottom row: Direct neighbors (down)
    ])

    # Iterate over each cell in the original matrix
    for i in range(rows):
        for j in range(cols):
            if np.isnan(matrix[i, j]):  # Only process NaN values
                # Extract the 3x3 neighborhood
                neighborhood = padded_matrix[i:i+3, j:j+3]

                # Apply weights to the neighborhood
                weighted_values = neighborhood * weights

                # Mask valid neighbors (non-NaN values)
                valid_mask = ~np.isnan(weighted_values)

                # Calculate the weighted average of valid neighbors
                if np.any(valid_mask):
                    filled_matrix[i, j] = np.sum(weighted_values[valid_mask]) / np.sum(weights[valid_mask])

    return filled_matrix
class SOEDClassifier:
    def __init__(self, 
                 mlp_hidden_layer_sizes=(100,), mlp_activation='relu', mlp_solver='adam',
                 mlp_alpha=0.0001, mlp_batch_size='auto', mlp_learning_rate='constant',
                 mlp_max_iter=200, mlp_tol=1e-4, mlp_verbose=False,
                 mlp_warm_start=False, mlp_n_iter_no_change=10, mlp_beta_1=0.9, mlp_beta_2=0.999, mlp_epsilon=1e-8,
                 som_x=10, som_y=10, som_input_len=None, som_sigma=1.0, som_learning_rate=0.5,
                 som_decay_function=None, som_neighborhood_function='gaussian', som_n_iter = 100,
                 random_state =None, soed_tune_percent=0.5):
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

        self.seod_tune_percent = soed_tune_percent

        #if self.som_random_seed is None:
        #    self.som_random_seed = np.random.randint(1000000)

        self.mlp = MLPRegressor(hidden_layer_sizes=self.mlp_hidden_layer_sizes, activation=self.mlp_activation,
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
        _multiplier = 0
        while _continue:
            _multiplier += 0.25
            X_som = np.column_stack((X, y*_multiplier,c*_multiplier))

            # Train the MiniSom
            self.som.train(X_som, self.som_n_iter)
            winner_coordinates = np.array([self.som.winner(x) for x in X_som])
            df = pd.DataFrame(np.column_stack((winner_coordinates,y,c)),columns=['X','Y','L','C0','C1'])
            pure_df = df.groupby(['X','Y','L']).size().unstack()
            is_pure = all(pure_df.loc[r].nunique(dropna=True) <= 1 for r in pure_df.index)

            _continue = not is_pure

        som_cost_df = df.copy(deep=True)
        som_cost_df['Cost_0'] = np.where(som_cost_df.L == 0.0,0.0,som_cost_df.C0)
        som_cost_df['Cost_1'] = np.where(som_cost_df.L == 1.0,0.0,som_cost_df.C1)

        som_cost_df = som_cost_df.groupby(['X','Y'])[['Cost_0','Cost_1']].sum()

        my_index = pd.MultiIndex.from_product([np.arange(self.som_x),np.arange(self.som_y)],names=['X','Y'])
        stage_df = pd.DataFrame(index = my_index, columns=['Cost_0','Cost_1'])
        stage_df.update(som_cost_df)

        som_cost_df = stage_df

        som_cost_0_df = som_cost_df['Cost_0'].unstack()
        som_cost_1_df = som_cost_df['Cost_1'].unstack()

        while som_cost_0_df.isna().sum().sum()>0:
            som_cost_0_df.loc[:,:]=fill_nan_with_weighted_neighbors(som_cost_0_df.values)

        while som_cost_1_df.isna().sum().sum()>0:
            som_cost_1_df.loc[:,:]=fill_nan_with_weighted_neighbors(som_cost_1_df.values)

        som_utility_df = pd.DataFrame({'Cost_0':som_cost_0_df.stack(), 'Cost_1':som_cost_1_df.stack()})
        som_utility_df['utility_0'] = som_utility_df['Cost_1']/som_utility_df[['Cost_0','Cost_1']].sum(axis=1)
        som_utility_df['utility_1'] = som_utility_df['Cost_0']/som_utility_df[['Cost_0','Cost_1']].sum(axis=1)

        som_decide_sr = pd.Series(np.where(som_utility_df.utility_1 > 0.5,1,0),index = som_utility_df.index)

        som_prob_df = df.groupby(['X','Y','L']).size().unstack().fillna(0.0)

        stage_df = pd.DataFrame(index = my_index, columns=[0.0,1.0])
        stage_df.update(som_prob_df)

        som_prob_df = stage_df

        som_prob_0_df = som_prob_df[0.0].unstack()
        som_prob_1_df = som_prob_df[1.0].unstack()

        while som_prob_0_df.isna().sum().sum()>0:
            som_prob_0_df.loc[:,:]=fill_nan_with_weighted_neighbors(som_prob_0_df.values)

        while som_prob_1_df.isna().sum().sum()>0:
            som_prob_1_df.loc[:,:]=fill_nan_with_weighted_neighbors(som_prob_1_df.values)

        som_porb_df = pd.DataFrame({'prob_0':som_prob_0_df.stack(), 'prob_1':som_prob_1_df.stack()})
        som_total_sum_sr = som_porb_df.sum(axis=1)
        som_porb_df['prob_0'] = som_porb_df['prob_0']/som_total_sum_sr
        som_porb_df['prob_1'] = som_porb_df['prob_1']/som_total_sum_sr



        y_mlp = (df[['X','Y']]+.5).values

        random_index = np.random.permutation(X.shape[0])
        i = int(round(X.shape[0]*self.seod_tune_percent))
        train_index = random_index[:i]
        tune_index = random_index[i+1:]

        X_train = X[train_index]
        y_mlp_train = y_mlp[train_index]
        y_train = y[train_index]
        c_train = c[train_index]

        X_tune = X[tune_index]
        y_mlp_tune = y_mlp[tune_index]
        y_tune = y[tune_index]
        c_tune = c[tune_index]


        # Train the MLPClassifier
        self.mlp.fit(X_train, y_mlp_train)
        tune_raw_predict = self.mlp.predict(X)
        tune_predict = (tune_raw_predict-0.5).round(0)
        tune_predict = np.where(tune_predict<0,0,tune_predict)
        tune_predict[:,0] = np.where(tune_predict[:,0]>self.som_x-1,0,tune_predict[:,0])
        tune_predict[:,1] = np.where(tune_predict[:,0]>self.som_y-1,0,tune_predict[:,1])

        df = pd.DataFrame(np.column_stack((tune_predict,y,c)),columns=['X','Y','L','C0','C1'])

        mlp_cost_df = df.copy(deep=True)
        mlp_cost_df['Cost_0'] = np.where(mlp_cost_df.L == 0.0,0.0,mlp_cost_df.C0)
        mlp_cost_df['Cost_1'] = np.where(mlp_cost_df.L == 1.0,0.0,mlp_cost_df.C1)

        mlp_cost_df = mlp_cost_df.groupby(['X','Y'])[['Cost_0','Cost_1']].sum()

        stage_df = pd.DataFrame(index = my_index, columns=['Cost_0','Cost_1'])
        stage_df.update(mlp_cost_df)

        mlp_cost_df = stage_df

        mlp_cost_0_df = mlp_cost_df['Cost_0'].unstack()
        mlp_cost_1_df = mlp_cost_df['Cost_1'].unstack()

        while mlp_cost_0_df.isna().sum().sum()>0:
            mlp_cost_0_df.loc[:,:]=fill_nan_with_weighted_neighbors(mlp_cost_0_df.values)

        while mlp_cost_1_df.isna().sum().sum()>0:
            mlp_cost_1_df.loc[:,:]=fill_nan_with_weighted_neighbors(mlp_cost_1_df.values)

        mlp_cost_0_df = (mlp_cost_0_df + som_cost_0_df)/2
        mlp_cost_1_df = (mlp_cost_1_df + som_cost_1_df)/2
        mlp_utility_df = pd.DataFrame({'Cost_0':mlp_cost_0_df.stack(), 'Cost_1':mlp_cost_1_df.stack()})
        mlp_utility_df['utility_0'] = mlp_utility_df['Cost_1']/mlp_utility_df[['Cost_0','Cost_1']].sum(axis=1)
        mlp_utility_df['utility_1'] = mlp_utility_df['Cost_0']/mlp_utility_df[['Cost_0','Cost_1']].sum(axis=1)





        cost_matrix_df = (mlp_cost_df.Cost_1 - mlp_cost_df.Cost_0).unstack()
        while cost_matrix_df.isna().sum().sum() >0:
            cost_matrix_df.loc[:,:]=fill_nan_with_weighted_neighbors(cost_matrix_df.values)

        cost_df = pd.DataFrame(np.column_stack((tune_predict,y,c)),columns=['X','Y','L','C0','C1'])
        cost_df['Cost_0'] = np.where(cost_df.L == 0.0,0.0,cost_df.C0)
        cost_df['Cost_1'] = np.where(cost_df.L == 1.0,0.0,cost_df.C1)


        decide_df = cost_df.groupby(['X','Y'])[['Cost_0','Cost_1']].sum()
        decide_df = (decide_df.Cost_1<decide_df.Cost_0).rename('class').reset_index().replace({True:1,False:0})

        prob_df = cost_df.groupby(['X','Y','L']).size().unstack().fillna(0.0)
        sum_sr = prob_df.sum(axis=1)

        prob_df[0.0] = prob_df[0.0] / sum_sr
        prob_df[1.0] = prob_df[1.0] / sum_sr

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
