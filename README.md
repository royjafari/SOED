# SOED
Self Organizing Error Driven Artificial Neural Network



# Publications
- Self-organizing and error driven (SOED) artificial neural network for smarter classifications https://academic.oup.com/jcde/article/4/4/282/5729001
- An optimum ANN-based breast cancer diagnosis: Bridging gaps between ANN learning and decision-making goals https://www.sciencedirect.com/science/article/abs/pii/S1568494618304484
- From in-situ monitoring toward high-throughput process control: cost-driven decision-making framework for laser-based additive manufacturing https://www.sciencedirect.com/science/article/abs/pii/S0278612518302929
- Optimum profit-driven churn decision making: innovative artificial neural networks in telecom industry https://link.springer.com/article/10.1007/s00521-020-04850-6
- Supervised or unsupervised learning? Investigating the role of pattern recognition assumptions in the success of binary predictive prescriptions https://www.sciencedirect.com/science/article/abs/pii/S0925231220319639

## How to work
```
pip install soed
```

```python

from soed import SOEDClassifier

soed = SOEDClassifier(mlp_max_iter=10000,som_x=10, som_y=10,som_input_len=X_train.shape[1])

```


## How to run project
```
virtualenv soed_dev
source soed_dev/bin/activate
pip install -r requirements.txt
```

```
jupyter notebook \
    --notebook-dir="./notebooks" \
    --ip=0.0.0.0 --port=3225
```

## At the time new py package installed
```
pip install <package>

pip freeze > requirements.txt
```



## for installation in local
```
pip install -e .
```


## build project
```
python setup.py sdist bdist_wheel
```

## Push project
```
twine upload dist/*
```

## Release pypi module
```
pip install twine
```
### Ensure you have pypi credential in
```
# nano ~/.pypirc
[pypi]
  username = __token__
  password = <Token>
```
```bash
./releaseNewVersionAndAddTag.sh
```