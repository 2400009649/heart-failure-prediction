# Import các thư viện cần thiết
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from lazypredict.Supervised import LazyClassifier
from imblearn.over_sampling import SMOTE
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
from skopt import BayesSearchCV
import joblib  # Để lưu mô hình
import warnings

# Tắt các cảnh báo
warnings.filterwarnings('ignore')

# Tải dữ liệu
def load_data(online=True):
    if online:
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00519/heart_failure_clinical_records_dataset.csv"
        data = pd.read_csv(url)
    else:
        path = "Dataset/heart_failure_clinical_records.xlsx"
        data = pd.read_excel(path)
    return data

# Tiền xử lý dữ liệu
def preprocess_data(data):
    data = data.replace('?', np.nan)
    data.dropna(inplace=True)
    data = data[(data.age >= 0) & (data.age <= 100)]
    X = data.drop(columns=['DEATH_EVENT'])
    y = data['DEATH_EVENT']
    return X, y

# Cân bằng dữ liệu
def balance_data(X_train, y_train):
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    return X_resampled, y_resampled

# Chuẩn hóa dữ liệu
def scale_data(X_train, X_test):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test

# Tự động thử nghiệm mô hình với LazyPredict
def compare_models(X_train, X_test, y_train, y_test):
    clf = LazyClassifier(verbose=0, ignore_warnings=True)
    models, predictions = clf.fit(X_train, X_test, y_train, y_test)
    return models

# Huấn luyện và tối ưu hóa tham số
def hyperopt_tuning(X_train, y_train, X_test, y_test):
    def objective(space):
        model = XGBClassifier(
            n_estimators=int(space['n_estimators']),
            learning_rate=space['learning_rate'],
            max_depth=int(space['max_depth']),
            subsample=space['subsample'],
            colsample_bytree=space['colsample_bytree']
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        f1 = f1_score(y_test, y_pred)
        return {'loss': -f1, 'status': STATUS_OK}

    space = {
        'n_estimators': hp.quniform('n_estimators', 50, 200, 10),
        'learning_rate': hp.uniform('learning_rate', 0.01, 0.3),
        'max_depth': hp.quniform('max_depth', 3, 10, 1),
        'subsample': hp.uniform('subsample', 0.6, 1),
        'colsample_bytree': hp.uniform('colsample_bytree', 0.6, 1)
    }

    trials = Trials()
    best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=50, trials=trials)
    return best

# Main
if __name__ == "__main__":
    data = load_data(online=True)
    X, y = preprocess_data(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Cân bằng dữ liệu
    X_train, y_train = balance_data(X_train, y_train)

    # Chuẩn hóa dữ liệu
    X_train, X_test = scale_data(X_train, X_test)

    # Tự động thử nghiệm mô hình
    print("Comparing models...")
    models_comparison = compare_models(X_train, X_test, y_train, y_test)
    print(models_comparison)

    # Tối ưu hóa tham số
    print("Optimizing XGBoost parameters...")
    best_params = hyperopt_tuning(X_train, y_train, X_test, y_test)
    print("Best Parameters:", best_params)

    # Huấn luyện mô hình XGBoost với tham số tối ưu
    best_model = XGBClassifier(
        n_estimators=int(best_params['n_estimators']),
        learning_rate=best_params['learning_rate'],
        max_depth=int(best_params['max_depth']),
        subsample=best_params['subsample'],
        colsample_bytree=best_params['colsample_bytree']
    )
    best_model.fit(X_train, y_train)

    # Lưu mô hình tối ưu
    joblib.dump(best_model, "xgboost_optimized.pkl")
    print("Optimized XGBoost model saved as xgboost_optimized.pkl")
