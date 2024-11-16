from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from load_titanic_data import load_titanic_data
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

class TitanicModel:
    def __init__(self):
        self.models = {}
        self.data = None
        self.scaler = None
        self.__load_data__()

    def train(self, model_type): #обучение модели
        if model_type == 'LogisticRegression':
            model = LogisticRegression()
            param_grid = {
                "penalty": ["l2", None], "C": np.logspace(-4, 4, 20),
                "solver": ["newton-cg", "lbfgs", "liblinear", "sag", "saga"], "max_iter": [100, 200, 300, 500],
            }
            n_iter = 20
        elif model_type == 'MLPClassifier':
            model = MLPClassifier()
            param_grid = {
                'hidden_layer_sizes': [(50,), (100,)],
                "activation": ["tanh", "relu"],
                "learning_rate": ["constant", "invscaling", "adaptive"],
            }
            n_iter = 5
        clf = RandomizedSearchCV(model, param_grid, random_state=0, scoring = 'f1_micro', verbose=3, n_iter=n_iter)
        clf.fit(self.data['x_scaled'], self.data['y'])
        self.models[model_type] = clf.best_estimator_

    def evaluate(self): #проверка насколько обученная модель хорошо обучилась
        pass

    def predict(self, input_values, model_type):
        predict_values = self.data['default_values'].copy()
        print(input_values)
        print(dir(input_values))
        input_values = input_values.__dict__
        for key, value in input_values.items():
            if type(value) in [int, float]:
                predict_values[key] = value
            else:
                if type(value) == str:
                    if key in ['who', 'deck']:
                        for s in predict_values.columns:
                            if s.startswith(key):
                                predict_values[s] = 0
                        predict_values[f"{key}_{value}"] = 1

        if 'sibsp' in input_values or 'parch' in input_values:
            rel = input_values.get('sibsp', 0) + input_values.get('parch', 0)
            if rel > 0:
                predict_values['alone'] = 0
            else:
                predict_values['alone'] = 1

        data_scaled = self.scaler.transform(predict_values)
        return self.models[model_type].predict_proba(data_scaled)[0][1]

    def __load_data__(self):
        x,y = load_titanic_data()
        self.scaler = StandardScaler().set_output(transform='pandas')
        x_scaled = self.scaler.fit_transform(x)
        default_values = x.mean()
        default_values = default_values.to_frame().transpose()
        self.data = {'x_scaled': x_scaled, 'y': y, 'default_values': default_values}

