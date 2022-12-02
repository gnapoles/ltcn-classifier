# Long-Term Cognitive Network for Pattern Classification

This repository introduces the Long-Term Cognitive Network (LTCN) model for structured pattern classification problems. This recurrent neural network incorporates a quasi-nonlinear reasoning rule that allows controlling the amount of non-linearity in the reasoning mechanism. Furthermore, this neural classifier uses a recurrence-aware decision model that evades the issues posed by the unique fixed point while introducing a deterministic learning algorithm to compute the tunable parameters. The simulations show that this classifier obtains competitive results when compared to state-of-the-art white and black-box models.

## Usage example

The syntax for the use of the LTCN classifier is compatible with `scikit-learn` library.

### Training

First create an LTCN object specifying the desired parameters. 

For our example with iris dataset, we will use the following paremters: `T=5` is the numner of iterations, `phi=0.9` is the amount of non-linearity during reasoning, `function='sigmoid'` is the activation function, `method='ridge'` is the learning algorithm and `alpha=1.0E-4` is the regularization penalty.

```python
model = LTCN(T=5, phi=0.9, function='sigmoid', method='ridge', alpha=1.0E-4)
```

For training a LTCN model simply call the fit method:

```python
model.fit(X_train, Y_train)
```

### Prediction

For predicting new data use the method predict:

```python
Y_pred = model.predict(X_test)
```

### Evaluation

Use the model within cross-validation (or any other suitable validation strategy from scikit-learn) for evaluating the performance. The example uses `5-fold-cv` and Kappa score for model selection:

```python
iris = datasets.load_iris(as_frame=True)

Y = pd.get_dummies(iris.target).values
X = iris.data.values

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
skf.get_n_splits(X, Y)

errors = []
for train_index, test_index in skf.split(X, iris.target):

    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]

    model = LTCN(T=5, phi=0.9, function='sigmoid', method='ridge', alpha=1.0E-4)
    model.fit(X_train, Y_train)
    
    Y_pred = model.predict(X_test)
    kappa = cohen_kappa_score(np.argmax(Y_test, axis=1), np.argmax(Y_pred, axis=1))
    errors.append(kappa)
    
print(sum(errors) / len(errors))
```

The minimal working example using Iris dataset is available as a [jupyter notebook](https://github.com/gnapoles/ltcn-classifier/blob/main/examples/example_iris.ipynb).

## Reference

Nápoles, G., Salgueiro, Y., Grau, I., & Espinosa, M. L. (2022). Recurrence-Aware Long-Term Cognitive Network for Explainable Pattern Classification. IEEE Transactions on Cybernetics. [paper](https://arxiv.org/pdf/2107.03423.pdf)
