import os
import sys
from dataclasses import dataclass

from catboost import CatBoostClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path =os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info('Splitting Training and test input Data')
            X_train, y_train, X_test, y_test =(
                train_array[:,:-1],
                train_array[:, -1],
                test_array[:,:-1],
                test_array[:, -1]
            )

            models  = {
                    'K-Neigbors Classifier': KNeighborsClassifier(),
                    'HistGradientBoosting Classifier': HistGradientBoostingClassifier(),
                    'GradientBoosting Classifier': GradientBoostingClassifier(),
                    'AdaBoost Classifier': AdaBoostClassifier(),
                    'RandomForest Classifier': RandomForestClassifier(),
                    'DecisionTree Classifier': DecisionTreeClassifier()
                        }
            
            params = {
                        "K-Neigbors Classifier": {
                            'n_neighbors': [3, 5, 7, 9, 11],
                            'weights': ['uniform', 'distance'],
                            'p': [1, 2]  # 1: Manhattan, 2: Euclidean
                        },
                        "HistGradientBoosting Classifier": {
                            'learning_rate': [0.01, 0.05, 0.1],
                            'max_depth': [3, 5, 7, None],
                            'max_iter': [50, 100, 200],
                            'min_samples_leaf': [10, 20, 30]
                        },
                        "GradientBoosting Classifier": {
                            'learning_rate': [0.01, 0.05, 0.1],
                            'n_estimators': [50, 100, 200],
                            'max_depth': [3, 5, 7],
                            'subsample': [0.7, 0.8, 0.9]
                        },
                        "AdaBoost Classifier": {
                            'n_estimators': [50, 100, 200],
                            'learning_rate': [0.01, 0.05, 0.1, 1.0]
                        },
                        "RandomForest Classifier": {
                            'n_estimators': [50, 100, 200],
                            'criterion': ['gini', 'entropy'],
                            'max_depth': [None, 5, 10, 20],
                            'max_features': ['sqrt', 'log2']
                        },
                        "DecisionTree Classifier": {
                            'criterion': ['gini', 'entropy'],
                            'splitter': ['best', 'random'],
                            'max_depth': [None, 5, 10, 20],
                            'max_features': ['sqrt', 'log2']
                        }
                    }

            
            model_report:dict = evaluate_models(X_train=X_train, y_train=y_train, X_test =X_test, 
                                                y_test  = y_test, models  =models, params = params)
            
             ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            ## To get best model name from dict

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score<0.0:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)

            accuracy = accuracy_score(y_test, predicted)

            print(f'{best_model_name}')
            return accuracy
        except Exception as e:
            raise CustomException(e, sys)

