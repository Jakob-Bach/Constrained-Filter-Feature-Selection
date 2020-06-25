"""Materials science pipeline

Prediction script for our case study in materials science.
"""


import sys

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from tqdm import tqdm
from xgboost import XGBRegressor

from ms_datasets import *

models = {
    'Dummy': DummyRegressor(strategy='mean'),
    'Decision tree': DecisionTreeRegressor(),
    'Linear regression': LinearRegression(),
    'xgboost': XGBRegressor(),
}

feature_set_sizes = [0.1, None]

print('Loading datasets ...')
sampled_voxel_dataset = prepare_sampled_voxel_data(delta_steps=20, subset='consecutive')
delta_voxel_dataset = prepare_delta_voxel_data(subset='consecutive')
prediction_problems = [
    predict_sampled_voxel_data_absolute(dataset=sampled_voxel_dataset),
    predict_sampled_voxel_data_relative(dataset=sampled_voxel_dataset),
    predict_delta_voxel_data_absolute(dataset=delta_voxel_dataset),
    predict_delta_voxel_data_relative(dataset=delta_voxel_dataset),
]

print('Predicting ...')
np.random.seed(25)
progress_bar = tqdm(total=len(models) * len(feature_set_sizes) * len(prediction_problems), file=sys.stdout)
results = []
for problem in prediction_problems:
    dataset = problem['dataset']
    dataset.dropna(subset=[problem['target']], inplace=True)
    max_train_time = dataset['time'].quantile(q=0.8)
    X_train = dataset[dataset['time'] <= max_train_time][problem['features']]
    y_train = dataset[dataset['time'] <= max_train_time][problem['target']]
    X_test = dataset[dataset['time'] > max_train_time][problem['features']]
    y_test = dataset[dataset['time'] > max_train_time][problem['target']]
    feature_qualities = [abs(X_train[feature].corr(y_train)) for feature in problem['features']]
    for num_features in feature_set_sizes:
        if num_features is None:  # no selection
            num_features = len(problem['features'])
        if num_features < 1:  # relative number of features
            num_features_rel = num_features
            num_features = round(num_features * len(problem['features']))  # turn absolute
        else:  # absolute number of features
            num_features_rel = num_features / len(problem['features'])
        if 1 <= num_features <= len(problem['features']):
            top_feature_idx = np.argsort(feature_qualities)[-num_features:]  # take last elements
            selected_features = [list(X_train)[idx] for idx in top_feature_idx]
        else:  # number of features does not make sense
            continue
        for model_name, model in models.items():
            model.fit(X_train[selected_features], y_train)
            pred_train = model.predict(X_train[selected_features])
            train_score = r2_score(y_true=y_train, y_pred=pred_train)
            pred_test = model.predict(X_test[selected_features])
            test_score = r2_score(y_true=y_test, y_pred=pred_test)
            results.append({'name': problem['name'], 'target': problem['target'],
                            'num_features': num_features, 'num_features_rel': num_features_rel,
                            'model': model_name, 'train_score': train_score, 'test_score': test_score})
            progress_bar.update()
progress_bar.close()
results = pd.DataFrame(results)
