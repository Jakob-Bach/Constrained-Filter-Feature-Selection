import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from ms_datasets import *

models = {
    'Dummy': DummyRegressor(strategy='mean'),
    'Decision tree': DecisionTreeRegressor(),
    'Linear regression': LinearRegression(),
    'xgboost': XGBRegressor(),
}

sampled_voxel_dataset = prepare_sampled_voxel_data(delta_steps=20, subset='consecutive')
delta_voxel_dataset = prepare_delta_voxel_data(subset='consecutive')
prediction_problems = [
    predict_sampled_voxel_data_absolute(dataset=sampled_voxel_dataset),
    predict_sampled_voxel_data_relative(dataset=sampled_voxel_dataset),
    predict_delta_voxel_data_absolute(dataset=delta_voxel_dataset),
    predict_delta_voxel_data_relative(dataset=delta_voxel_dataset),
]

np.random.seed(25)
results = []
for problem in prediction_problems:
    dataset = problem['dataset']
    dataset.dropna(subset=[problem['target']], inplace=True)
    max_train_time = dataset['time'].quantile(q=0.8)
    X_train = dataset[dataset['time'] <= max_train_time][problem['features']]
    y_train = dataset[dataset['time'] <= max_train_time][problem['target']]
    X_test = dataset[dataset['time'] > max_train_time][problem['features']]
    y_test = dataset[dataset['time'] > max_train_time][problem['target']]
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        pred_train = model.predict(X_train)
        train_score = r2_score(y_true=y_train, y_pred=pred_train)
        pred_test = model.predict(X_test)
        test_score = r2_score(y_true=y_test, y_pred=pred_test)
        results.append({'name': problem['name'], 'target': problem['target'], 'model': model_name,
                        'train_score': train_score, 'test_score': test_score})
results = pd.DataFrame(results)

for scenario in results['name'].unique():
    results[results['name'] == scenario].set_index(['target', 'model']).plot(
        kind='bar', ylim=(-0.2, 1.2), title='Scenario: '+ scenario)
    plt.show()
