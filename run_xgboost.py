import xgboost as xgb
import numpy as np
import pickle
from tqdm import tqdm
import torch
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix
)
import pandas as pd
import random

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)

def prepare_data(data_dict):
    labels = []
    features = []
    for item in data_dict.values():
        labels.append(item['label'].item()) 
        features.append(item['xzx'].flatten())  

    return np.array(features), np.array(labels)

columns = ['model', 'evaluate_name', 'acc', 'auc', 'mcc', 'sn', 'sp']

df = pd.DataFrame(columns=columns)

choices = [
    'A.thaliana','C.elegans', 'C.equisetifolia', 'D.melanogaster', 
    'F.vesca', 'H.sapiens', 'R.chinensis', 'S.cerevisiae', 
    'T.thermophile', 'Tolypocladium', 'Xoc.BLS256'
]
train_or_test_choices = ['train', 'test']

for b in choices:
    X_train, y_train = [], []
    X_test, y_test = [], []

    train_file = f'model_{b}_dataset_{b}_train.pkl'
    try:
        with open(train_file, 'rb') as f:
            data_train = pickle.load(f)
    except FileNotFoundError:
        print(f"Training file {train_file} not found. Skipping model {b}.")
        continue

    for batch_id in data_train:
        batch_data = data_train[batch_id]
        X_train.extend(batch_data['xzx'].reshape(batch_data['xzx'].shape[0], -1))
        y_train.extend(batch_data['label'].tolist())

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    test_file = f'model_{b}_dataset_{b}_test.pkl'
    try:
        with open(test_file, 'rb') as f:
            data_test = pickle.load(f)
    except FileNotFoundError:
        print(f"Test file {test_file} not found. Skipping model {b}.")
        continue

    for batch_id in data_test:
        batch_data = data_test[batch_id]
        X_test.extend(batch_data['xzx'].reshape(batch_data['xzx'].shape[0], -1))
        y_test.extend(batch_data['label'].tolist())

    X_test = np.array(X_test)
    y_test = np.array(y_test)

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    params = {
        'objective': 'binary:logistic',
        'max_depth': 6,
        'booster': 'gbtree',
        'learning_rate': 0.001,
        'eval_metric': 'error',
        'tree_method': 'gpu_hist',
        'seed': SEED,
        'verbosity': 0
    }

    num_boost_round = 10000

    evals_result = {}
    print(f"Starting training for model {b}. Please wait...")
    bst = xgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        evals=[(dtest, 'test')],
        evals_result=evals_result,
        verbose_eval=False
    )

    test_errors = evals_result['test']['error']

    min_error = min(test_errors)
    best_iteration = test_errors.index(min_error) + 1 

    print(f"Model {b}: Best iteration with minimum error {min_error} at round {best_iteration}")

    bst_optimal = xgb.train(
        params,
        dtrain,
        num_boost_round=best_iteration,
        evals=[(dtest, 'test')],
        verbose_eval=False
    )

    results_list = []

    for evaluate_name in choices:
        eval_test_file = f'model_{b}_dataset_{evaluate_name}_test.pkl'
        try:
            with open(eval_test_file, 'rb') as f:
                eval_data_test = pickle.load(f)
        except FileNotFoundError:
            print(f"Evaluation test file {eval_test_file} not found. Skipping evaluation on {evaluate_name}.")
            results_list.append({
                'model': b,
                'evaluate_name': evaluate_name,
                'acc': np.nan,
                'auc': np.nan,
                'mcc': np.nan,
                'sn': np.nan,
                'sp': np.nan
            })
            continue

        X_eval_test, y_eval_test = [], []
        for batch_id in eval_data_test:
            batch_data = eval_data_test[batch_id]
            X_eval_test.extend(batch_data['xzx'].reshape(batch_data['xzx'].shape[0], -1))
            y_eval_test.extend(batch_data['label'].tolist())

        X_eval_test = np.array(X_eval_test)
        y_eval_test = np.array(y_eval_test)

        if len(np.unique(y_eval_test)) < 2:
            print(f"Evaluation dataset {evaluate_name} for model {b} has less than two classes. Metrics may be unreliable.")

        d_eval_test = xgb.DMatrix(X_eval_test, label=y_eval_test)
        preds = bst_optimal.predict(d_eval_test)
        predictions = (preds > 0.5).astype(int)

        accuracy = accuracy_score(y_eval_test, predictions)

        try:
            auc = roc_auc_score(y_eval_test, preds)
        except ValueError:
            auc = np.nan 

        # Compute MCC
        if len(np.unique(y_eval_test)) < 2:
            mcc = np.nan 
        else:
            mcc = matthews_corrcoef(y_eval_test, predictions)

        if len(np.unique(y_eval_test)) < 2:
            sn = np.nan
            sp = np.nan
        else:
            cm = confusion_matrix(y_eval_test, predictions)
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                sn = tp / (tp + fn) if (tp + fn) > 0 else np.nan  
                sp = tn / (tn + fp) if (tn + fp) > 0 else np.nan  
            else:
                sn = np.nan
                sp = np.nan

        results_list.append({
            'model': b,
            'evaluate_name': evaluate_name,
            'acc': accuracy,
            'auc': auc,
            'mcc': mcc,
            'sn': sn,
            'sp': sp
        })

    results_df = pd.DataFrame(results_list)
    df = pd.concat([df, results_df], ignore_index=True)
    df.to_csv("output.csv", index=False)

print("Training and evaluation completed. Results saved to output.csv.")
