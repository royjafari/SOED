import time

time.sleep(5)

import numpy as np

from sklearn.metrics import precision_score, recall_score, roc_auc_score, accuracy_score

X = np.random.random([10000,10])
y = np.random.choice([0,1],10000)
c = np.random.random([10000,2])


from soed import SOEDClassifier

from sklearn.datasets import load_iris

# Load dataset
data = load_iris()
X = data.data
y = data.target  # Binary target

y = np.where(y==2,0,y)

X_standard = (X-X.min(axis=0))/(X.max(axis=0) - X.min(axis=0))

random_index = np.random.permutation(X.shape[0])
i = int(round(X.shape[0]*0.5))
train_index = random_index[:i]
test_index = random_index[i+1:]

X_train = X_standard[train_index]
X_test = X_standard[test_index]

y_train = y[train_index]
y_test = y[test_index]


c0 = np.where(y_train==0,0,20)
c1 = np.where(y_train==1,0,1)

c = np.column_stack((c0,c1))



soed = SOEDClassifier(mlp_max_iter=10000,som_x=10, som_y=10,som_input_len=X_train.shape[1])

soed.fit(X_train,y_train,c)

y_proba = soed.predict_proba(X_test)
y_pred = soed.predict(X_test)

y_util = soed.predict_util(X_test)
y_decide = soed.decide(X_test)



print(recall_score(y_test,y_pred))
print(precision_score(y_test,y_pred))
print(accuracy_score(y_test,y_pred))
print(roc_auc_score(y_test,y_proba[:,1]))



print(soed)
