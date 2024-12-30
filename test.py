import time

time.sleep(5)

import numpy as np

X = np.random.random([10000,10])
y = np.random.choice([0,1],10000)
c = np.random.random([10000,2])


from main import SOEDClassifier

soed = SOEDClassifier(mlp_max_iter=10000,som_x=7, som_y=7,som_input_len=X.shape[1])

soed.fit(X,y)



print(soed)