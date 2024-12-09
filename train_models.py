# Import các thư viện
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
    # Xử lý giá trị thiếu
    data = data.replace('?', np.nan)
    data.dropna(inplace=True)
    
    # Xử lý bất thường
    data = data[(data.age >= 0) & (data.age <= 100)]
    
    # Chia tập dữ liệu
    X = data.drop(columns=['DEATH_EVENT'])
    y = data['DEATH_EVENT']
    
    return X, y

# Chuẩn hóa dữ liệu
def scale_data(X_train, X_test):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test

# Huấn luyện và đánh giá mô hình
def train_and_save_models(X_train, X_test, y_train, y_test):
    models = {
        "LogisticRegression": LogisticRegression(),
        "DecisionTree": DecisionTreeClassifier(),
        "RandomForest": RandomForestClassifier(),
        "XGBoost": XGBClassifier(),
        "LightGBM": LGBMClassifier()
    }
    
    results = {}
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Lưu mô hình
        model_filename = f"{name.lower()}.pkl"
        joblib.dump(model, model_filename)
        print(f"Saved {name} model to {model_filename}")
        
        # Lưu kết quả
        results[name] = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "F1 Score": f1_score(y_test, y_pred),
            "ROC AUC": roc_auc_score(y_test, y_proba) if y_proba is not None else None,
            "Classification Report": classification_report(y_test, y_pred)
        }
    
    return results

# Hiển thị kết quả
def display_results(results):
    for name, metrics in results.items():
        print(f"Model: {name}")
        for metric, value in metrics.items():
            print(f"{metric}: {value}")
        print("\n" + "="*30 + "\n")

# Main
if __name__ == "__main__":
    # Tải và xử lý dữ liệu
    data = load_data(online=True)
    X, y = preprocess_data(data)
    
    # Chia tập dữ liệu
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Chuẩn hóa dữ liệu
    X_train, X_test = scale_data(X_train, X_test)
    
    # Huấn luyện và lưu mô hình
    results = train_and_save_models(X_train, X_test, y_train, y_test)
    
    # Hiển thị kết quả
    display_results(results)
