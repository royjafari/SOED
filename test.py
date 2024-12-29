import time

time.sleep(5)

import numpy as np

X = np.random.random([1000,10])
y = np.random.choice([0,1],1000)


from main import SOEDClassifier

soed = SOEDClassifier(som_x=3, som_y=3,som_input_len=X.shape[1])

soed.fit(X,y)



print(soed)