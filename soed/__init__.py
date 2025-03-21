import numpy as np
import pandas as pd
from minisom import MiniSom
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor

model = MultiOutputRegressor(XGBRegressor())


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
                 mlp_max_iter=1000, mlp_tol=1e-4, mlp_verbose=False,
                 mlp_warm_start=False, mlp_n_iter_no_change=10, mlp_beta_1=0.9, mlp_beta_2=0.999, mlp_epsilon=1e-8,
                 som_x=10, som_y=10, som_input_len=None, som_sigma=3, som_learning_rate=0.25,
                 som_decay_function=None, som_neighborhood_function='gaussian', som_n_iter = 1000,
                 xgb_n_estimators=500, xgb_max_depth=100, xgb_learning_rate=0.1, xgb_subsample=0.8, xgb_colsample_bytree=0.8,
                 xgb_gamma=0, xgb_reg_alpha=0, xgb_reg_lambda=1, xgb_random_state=None,
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

        #assert type(som_input_len)==int, 'som_input_len needs to be integer.'
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
        self.som_sigma = som_sigma
        self.som_learning_rate = som_learning_rate
        self.som_decay_function = som_decay_function
        self.som_neighborhood_function = som_neighborhood_function
        self.som_random_seed = random_state
        self.som_n_iter = som_n_iter

        # XGBoost Parameters
        self.xgb_n_estimators = xgb_n_estimators
        self.xgb_max_depth = xgb_max_depth
        self.xgb_learning_rate = xgb_learning_rate
        self.xgb_subsample = xgb_subsample
        self.xgb_colsample_bytree = xgb_colsample_bytree
        self.xgb_gamma = xgb_gamma
        self.xgb_reg_alpha = xgb_reg_alpha
        self.xgb_reg_lambda = xgb_reg_lambda
        self.xgb_random_state = xgb_random_state

        self.seod_tune_percent = soed_tune_percent


        self.mlp = MLPRegressor(hidden_layer_sizes=self.mlp_hidden_layer_sizes, activation=self.mlp_activation,
                                 solver=self.mlp_solver, alpha=self.mlp_alpha, batch_size=self.mlp_batch_size,
                                 learning_rate=self.mlp_learning_rate, max_iter=self.mlp_max_iter,
                                 random_state=self.mlp_random_state, tol=self.mlp_tol, verbose=self.mlp_verbose,
                                 warm_start=self.mlp_warm_start, n_iter_no_change=self.mlp_n_iter_no_change,
                                 beta_1=self.mlp_beta_1, beta_2=self.mlp_beta_2, epsilon=self.mlp_epsilon)


        params = {
            "objective": "reg:squarederror",  # For regression task
            "booster": "gbtree",  # Tree boosting
            "learning_rate": self.xgb_learning_rate,  # Step size at each iteration
            "max_depth": self.xgb_max_depth,  # Maximum depth of a tree
            "n_estimators": self.xgb_n_estimators,  # Number of trees
            "subsample": self.xgb_subsample,  # Subsample ratio of the training set
            "colsample_bytree": self.xgb_colsample_bytree,  # Subsample ratio of columns when constructing each tree
            "gamma": self.xgb_gamma,  # Minimum loss reduction required to make a further partition
            "reg_alpha": self.xgb_reg_alpha,  # L1 regularization term on weights
            "reg_lambda": self.xgb_reg_lambda,  # L2 regularization term on weights
        }

        xgb_model = XGBRegressor(**params)
        self.xgb = MultiOutputRegressor(xgb_model)

        self.decide_sr = None
        self.predict_sr = None
        self.prob_df = None
        self.utility_df = None
        self.is_fitted = False
        print('version 1.0.9')

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
            c_standard = c
        else:
            c_standard = (c - c.min())/(c.max()-c.min())

        assert X.shape[0] == y.shape[0], 'X and y must be of the same length.'
        assert X.shape[0] == c.shape[0], 'X and c must be of the same length.'
        assert c.shape[1] == 2, 'c must have two columns.'
        assert np.isin(y, [0, 1]).all(), 'y can only have binary values.'

        X_df = pd.DataFrame(X)

        corr_matrix = X_df.corr().abs()

        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.9)]

        rang_df = (X_df.max()-X_df.min())
        to_drop2 = rang_df[rang_df == 0].index.tolist()
        to_drop.extend(to_drop2)

        X_select = X_df.drop(columns=to_drop).values

        X_select_standard = (X_select - X_select.min(axis=0))/(X_select.max(axis=0) - X_select.min(axis=0))


        #X_standard = np.nan_to_num((X - X.mean(axis=0))/X.std(axis=0),0.0)
        #pca = PCA(n_components=X.shape[1])  # Reduce to 7 dimensions
        #X_pca = pd.DataFrame(pca.fit_transform(X_standard),columns = [f'PC{i}' for i in range(1,X.shape[1]+1)])

        #explained_variance_ratio = pca.explained_variance_ratio_
        #cumulative_explained_variance = np.cumsum(explained_variance_ratio)

        #n_dimensions = min(
        #    (cumulative_explained_variance<0.9).sum()+1,
        #    X.shape[0]
        #)
        #X_pca_select = X_pca[[f'PC{i}' for i in range(1,n_dimensions+1)]]



        self.som = MiniSom(self.som_x, self.som_y, X_select.shape[1]+3, sigma=self.som_sigma,
                           learning_rate=self.som_learning_rate,
                           neighborhood_function=self.som_neighborhood_function, random_seed=self.som_random_seed)

        _continue = True
        _multiplier = -0.125
        while _continue:
            _multiplier += 0.125
            X_som = np.column_stack((X_select_standard, y*_multiplier,c_standard*_multiplier))

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
        stage_df = pd.DataFrame(0.01,index = my_index, columns=['Cost_0','Cost_1'])
        stage_df.update(som_cost_df)

        som_cost_df = stage_df

        som_cost_0_df = som_cost_df['Cost_0'].unstack()
        som_cost_1_df = som_cost_df['Cost_1'].unstack()

        while som_cost_0_df.isna().sum().sum()>0:
            som_cost_0_df.loc[:,:]=fill_nan_with_weighted_neighbors(som_cost_0_df.values)

        while som_cost_1_df.isna().sum().sum()>0:
            som_cost_1_df.loc[:,:]=fill_nan_with_weighted_neighbors(som_cost_1_df.values)

        som_utility_df = pd.DataFrame({'Cost_0':som_cost_0_df.stack(), 'Cost_1':som_cost_1_df.stack()})
        BM = (som_utility_df.Cost_0 == 0) & (som_utility_df.Cost_1==0)
        if BM.sum()>0:
            for i in som_utility_df[BM].index:
                som_utility_df.loc[i,'Cost_0'] = 0.000001
                som_utility_df.loc[i,'Cost_1'] = 0.000001

        som_utility_df['utility_0'] = som_utility_df['Cost_1']/som_utility_df[['Cost_0','Cost_1']].sum(axis=1)
        som_utility_df['utility_1'] = som_utility_df['Cost_0']/som_utility_df[['Cost_0','Cost_1']].sum(axis=1)

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

        y_supervised = (df[['X','Y']]+0.5).values

        random_index = np.random.permutation(X.shape[0])
        i = int(round(X.shape[0]*self.seod_tune_percent))
        train_index = random_index[:i]

        X_train = X[train_index]
        y_supervised_train = y_supervised[train_index]


        # Train the MLPClassifier
        self.xgb.fit(X_train, y_supervised_train)
        tune_raw_predict = self.xgb.predict(X)
        tune_predict = (tune_raw_predict-0.5).round(0)
        tune_predict = np.where(tune_predict<0,0,tune_predict)
        tune_predict[:,0] = np.where(tune_predict[:,0]>self.som_x-1,self.som_x-1,tune_predict[:,0])
        tune_predict[:,1] = np.where(tune_predict[:,1]>self.som_y-1,self.som_y-1,tune_predict[:,1])

        df = pd.DataFrame(np.column_stack((tune_predict,y,c)),columns=['X','Y','L','C0','C1'])

        xgb_cost_df = df.copy(deep=True)
        xgb_cost_df['Cost_0'] = np.where(xgb_cost_df.L == 0.0,0.0,xgb_cost_df.C0)
        xgb_cost_df['Cost_1'] = np.where(xgb_cost_df.L == 1.0,0.0,xgb_cost_df.C1)

        xgb_cost_df = xgb_cost_df.groupby(['X','Y'])[['Cost_0','Cost_1']].sum()

        stage_df = pd.DataFrame(index = my_index, columns=['Cost_0','Cost_1'])
        stage_df.update(xgb_cost_df)

        xgb_cost_df = stage_df

        xgb_cost_0_df = xgb_cost_df['Cost_0'].unstack()
        xgb_cost_1_df = xgb_cost_df['Cost_1'].unstack()

        while xgb_cost_0_df.isna().sum().sum()>0:
            xgb_cost_0_df.loc[:,:]=fill_nan_with_weighted_neighbors(xgb_cost_0_df.values)

        while xgb_cost_1_df.isna().sum().sum()>0:
            xgb_cost_1_df.loc[:,:]=fill_nan_with_weighted_neighbors(xgb_cost_1_df.values)

        xgb_cost_0_df = (xgb_cost_0_df + som_cost_0_df)/2
        xgb_cost_1_df = (xgb_cost_1_df + som_cost_1_df)/2
        xgb_utility_df = pd.DataFrame({'Cost_0':xgb_cost_0_df.stack(), 'Cost_1':xgb_cost_1_df.stack()})
        BM = (xgb_utility_df.Cost_0 == 0) & (xgb_utility_df.Cost_1==0)
        if BM.sum()>0:
            for i in xgb_utility_df[BM].index:
                xgb_utility_df.loc[i,'Cost_0'] = 0.000001
                xgb_utility_df.loc[i,'Cost_1'] = 0.000001
        xgb_utility_df['utility_0'] = xgb_utility_df['Cost_1']/xgb_utility_df[['Cost_0','Cost_1']].sum(axis=1)
        xgb_utility_df['utility_1'] = xgb_utility_df['Cost_0']/xgb_utility_df[['Cost_0','Cost_1']].sum(axis=1)

        self.utility_df = xgb_utility_df

        xgb_decide_sr = pd.Series(
            np.where(xgb_utility_df.utility_1 > xgb_utility_df.utility_0,1,0),
            index = xgb_utility_df.index)

        self.decide_sr = xgb_decide_sr

        xgb_prob_df = df.groupby(['X','Y','L']).size().unstack().fillna(0.0)

        stage_df = pd.DataFrame(index = my_index, columns=[0.0,1.0])
        stage_df.update(xgb_prob_df)

        xgb_prob_df = stage_df

        xgb_prob_0_df = xgb_prob_df[0.0].unstack()
        xgb_prob_1_df = xgb_prob_df[1.0].unstack()

        while xgb_prob_0_df.isna().sum().sum()>0:
            xgb_prob_0_df.loc[:,:]=fill_nan_with_weighted_neighbors(xgb_prob_0_df.values)

        while xgb_prob_1_df.isna().sum().sum()>0:
            xgb_prob_1_df.loc[:,:]=fill_nan_with_weighted_neighbors(xgb_prob_1_df.values)

        xgb_prob_1_df = (xgb_prob_1_df+som_prob_1_df)/2
        xgb_prob_0_df = (xgb_prob_0_df+som_prob_0_df)/2

        xgb_porb_df = pd.DataFrame({'prob_0':xgb_prob_0_df.stack(), 'prob_1':xgb_prob_1_df.stack()})
        xgb_total_sum_sr = xgb_porb_df.sum(axis=1)
        xgb_porb_df['prob_0'] = xgb_porb_df['prob_0']/xgb_total_sum_sr
        xgb_porb_df['prob_1'] = xgb_porb_df['prob_1']/xgb_total_sum_sr

        xgb_predict_sr = pd.Series(
            np.where(xgb_porb_df.prob_1 > xgb_porb_df.prob_0,1,0),
            index = xgb_utility_df.index)

        self.predict_sr = xgb_predict_sr

        self.prob_df = xgb_porb_df

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

        # Predict using the trained MLPClassifier
        y_pred = self.xgb.predict(X)
        y_pred = (y_pred-0.5).round(0)
        y_pred = np.where(y_pred<0,0,y_pred)
        y_pred[:,0] = np.where(y_pred[:,0]>self.som_x-1,self.som_x-1,y_pred[:,0])
        y_pred[:,1] = np.where(y_pred[:,1]>self.som_y-1,self.som_y-1,y_pred[:,1])

        output = (
            self.predict_sr
            .loc[
                [(int(y_pred[i][0]),(int(y_pred[i][1]))) for i in range(y_pred.shape[0])]
                ]
            .reset_index(drop=True)
            .values
        )

        return output

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

        y_pred = self.xgb.predict(X)
        y_pred = (y_pred-0.5).round(0)
        y_pred = np.where(y_pred<0,0,y_pred)
        y_pred[:,0] = np.where(y_pred[:,0]>self.som_x-1,self.som_x-1,y_pred[:,0])
        y_pred[:,1] = np.where(y_pred[:,1]>self.som_y-1,self.som_y-1,y_pred[:,1])

        proba = (
            self.prob_df
            .loc[
                [(int(y_pred[i][0]),(int(y_pred[i][1]))) for i in range(y_pred.shape[0])]
                ]
            .reset_index(drop=True)
            .values
        )


        return proba


    def decide_util(self, X):
        """
        Utility estimates for samples in X.

        Parameters:
        - X: array-like of shape (n_samples, n_features)
            The input data.

        Returns:
        - util: array-like of shape (n_samples, n_classes)
            Utilities estimates.
        """
        if not self.is_fitted:
            raise ValueError("The model has not been fitted yet.")

        y_pred = self.xgb.predict(X)
        y_pred = (y_pred-0.5).round(0)
        y_pred = np.where(y_pred<0,0,y_pred)
        y_pred[:,0] = np.where(y_pred[:,0]>self.som_x-1,self.som_x-1,y_pred[:,0])
        y_pred[:,1] = np.where(y_pred[:,1]>self.som_y-1,self.som_y-1,y_pred[:,1])

        util = (
            self.utility_df[['utility_0','utility_1']]
            .loc[
                [(int(y_pred[i][0]),(int(y_pred[i][1]))) for i in range(y_pred.shape[0])]
                ]
            .reset_index(drop=True)
            .values
        )

        return util

    def decide(self, X):
        """
        Predict class labels for samples in X.

        Parameters:
        - X: array-like of shape (n_samples, n_features)
            The input data.

        Returns:
        - y_pred: array-like of shape (n_samples,)
            Make a decision.
        """
        if not self.is_fitted:
            raise ValueError("The model has not been fitted yet.")

        # Predict using the trained MLPClassifier
        y_pred = self.xgb.predict(X)
        y_pred = (y_pred-0.5).round(0)
        y_pred = np.where(y_pred<0,0,y_pred)
        y_pred[:,0] = np.where(y_pred[:,0]>self.som_x-1,self.som_x-1,y_pred[:,0])
        y_pred[:,1] = np.where(y_pred[:,1]>self.som_y-1,self.som_y-1,y_pred[:,1])

        output = (
            self.decide_sr
            .loc[
                [(int(y_pred[i][0]),(int(y_pred[i][1]))) for i in range(y_pred.shape[0])]
                ]
            .reset_index(drop=True)
            .values
        )

        return output
