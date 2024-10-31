from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from load_titanic_data import load_titanic_data
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

class TitanicModel:
    def __init__(self):
        self.models = None
        self.data = None
        self.scaler = None
        self.__load_data__()

    def train(self, model_type): #обучение модели
        if model_type == 'Logistic Regression':
            model = LogisticRegression()
            param_grid = {
                "penalty": ["l1", "l2", "elasticnet", "none"], "C": np.logspace(-4, 4, 20),
                "solver": ["newton-cg", "lbfgs", "liblinear", "sag", "saga"], "max_iter": [100, 200, 300, 500],
            }
        elif model_type == 'MLP Classifier':
            model = MLPClassifier()
            param_grid = {
                'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 100)],
                "activation": ["tanh", "relu"],
                "learning_rate": ["constant", "invscaling", "adaptive"],
            }
        clf = RandomizedSearchCV(model, param_grid, random_state=0, scoring = 'f1_micro')
        search = clf.fit(self.data['x_scaled'], self.data['y'])
        self.models[model_type] = search.best_estimator_

    def evaluate(self): #проверка насколько обученная модель хорошо обучилась
        pass

    def predict(self, input_values, model_type):
        predict_values = self.data['default_values'].copy()
        for key, value in input_values.items():
            predict_values[key] = value

        data_scaled = self.scaler.transform(predict_values)
        return self.models[model_type].predict_proba(data_scaled)[0][1]

    def __load_data__(self):
        x,y = load_titanic_data()
        self.scaler = StandardScaler().set_output(transform='pandas')
        x_scaled = self.scaler.fit_transform(x)
        default_values = x.mean()
        default_values = default_values.to_frame().transpose()
        self.data = {'x_scaled': x_scaled, 'y': y, 'default_values': default_values}

