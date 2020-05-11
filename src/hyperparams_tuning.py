import numpy as np
from sklearn.model_selection import cross_val_score
from skopt.utils import use_named_args
from skopt import forest_minimize
from skopt.plots import plot_convergence
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

def tune_hyperparams(model, space, X, y):
    """Tune hyper-parameters of a given model.
    
    Use mean cross-validation accuracy score to evaluate model.
    
    Parameters
    ----------
    model : instance
        a sklearn classifer
        
    space : list of skopt space
        a search space of model hyper-parameters
    
    X : pandas dataframe or numpy.ndarray
        features used for tuning
        
    y : numpy array
        target used for tuning
        
    Returns
    -------
    best_params : dict
        a dictionary containing best parameters
        
    best_score : float
        best mean cross validation accuracy score 
        
        
    Examples
    --------
    >>> from sklearn.datasets import load_breast_cancer
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from skopt.space import Real, Integer, Categorical
    >>> from src.hyperparams_tuning import tune_hyperparams
    >>> X, y = load_breast_cancer(return_X_y=True)
    >>> model = RandomForestClassifier(random_state=0)
    >>> space = [Integer(2, 20, name='max_depth'),
    ...          Integer(2, 20, name='max_leaf_nodes')]
    >>> best_params, best_score = tune_hyperparams(model, space, X, y)
    """
    @use_named_args(space)
    def objective(**params):
        model.set_params(**params)
        return -np.mean(cross_val_score(model, X, y, scoring='accuracy', cv=5))

    results = forest_minimize(objective, space, n_calls=10, random_state=0)
    
    best_params = dict()
    for i in range(len(space)):
        best_params[space[i].name] = results.x[i]
    
    best_score = -results.fun    
    #plot_convergence(results)
    #plt.show()
    return best_params, best_score
