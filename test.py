import time

time.sleep(5)

import numpy as np

X = np.random.random([100000,10])
y = np.random.choice([0,1],100000)
c = np.random.random([100000,2])


from main import SOEDClassifier

soed = SOEDClassifier(mlp_max_iter=10000,som_x=15, som_y=15,som_input_len=X.shape[1])

soed.fit(X,y,c)



print(soed)