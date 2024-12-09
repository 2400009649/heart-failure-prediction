import streamlit as st
import pandas as pd
import joblib
from translations import translations

# Load các mô hình
models = {
    "LogisticRegression": joblib.load("logisticregression.pkl"),
    "DecisionTree": joblib.load("decisiontree.pkl"),
    "RandomForest": joblib.load("randomforest.pkl"),
    "XGBoost": joblib.load("xgboost.pkl"),
    "LightGBM": joblib.load("lightgbm.pkl"),
}

# Chọn ngôn ngữ
language = st.sidebar.selectbox("Choose Language / Chọn ngôn ngữ", ["English", "Vietnamese"])
t = translations[language]  # Lấy bản dịch tương ứng

# Tiêu đề ứng dụng
st.title(t["title"])

# Form nhập liệu
st.header(t["enter_data"])

# Hiển thị tham số với giải thích gọn gàng
with st.expander("Click here for parameter explanations / Bấm để xem giải thích các tham số"):
    st.write(f"**{t['age']}**: {t['age_desc']}")
    st.write(f"**{t['anaemia']}**: {t['anaemia_desc']}")
    st.write(f"**{t['creatinine_phosphokinase']}**: {t['creatinine_phosphokinase_desc']}")
    st.write(f"**{t['diabetes']}**: {t['diabetes_desc']}")
    st.write(f"**{t['ejection_fraction']}**: {t['ejection_fraction_desc']}")
    st.write(f"**{t['high_blood_pressure']}**: {t['high_blood_pressure_desc']}")
    st.write(f"**{t['platelets']}**: {t['platelets_desc']}")
    st.write(f"**{t['serum_creatinine']}**: {t['serum_creatinine_desc']}")
    st.write(f"**{t['serum_sodium']}**: {t['serum_sodium_desc']}")
    st.write(f"**{t['sex']}**: {t['sex_desc']}")
    st.write(f"**{t['smoking']}**: {t['smoking_desc']}")
    st.write(f"**{t['time']}**: {t['time_desc']}")

# Các field nhập liệu chính
age = st.number_input(t["age"], min_value=0, max_value=120, value=60, help=t["age_desc"])
anaemia = st.selectbox(t["anaemia"], [0, 1], help=t["anaemia_desc"])
creatinine_phosphokinase = st.number_input(t["creatinine_phosphokinase"], min_value=0, max_value=10000, value=582, help=t["creatinine_phosphokinase_desc"])
diabetes = st.selectbox(t["diabetes"], [0, 1], help=t["diabetes_desc"])
ejection_fraction = st.number_input(t["ejection_fraction"], min_value=0, max_value=100, value=30, help=t["ejection_fraction_desc"])
high_blood_pressure = st.selectbox(t["high_blood_pressure"], [0, 1], help=t["high_blood_pressure_desc"])
platelets = st.number_input(t["platelets"], min_value=0.0, max_value=1000000.0, value=265000.0, help=t["platelets_desc"])
serum_creatinine = st.number_input(t["serum_creatinine"], min_value=0.0, max_value=20.0, value=1.2, help=t["serum_creatinine_desc"])
serum_sodium = st.number_input(t["serum_sodium"], min_value=100, max_value=200, value=140, help=t["serum_sodium_desc"])
sex = st.selectbox(t["sex"], [0, 1], help=t["sex_desc"])
smoking = st.selectbox(t["smoking"], [0, 1], help=t["smoking_desc"])
time = st.number_input(t["time"], min_value=0, max_value=1000, value=4, help=t["time_desc"])

# Dự đoán khi nhấn nút
if st.button(t["predict"]):
    # Chuẩn bị dữ liệu đầu vào
    input_data = {
        "age": age,
        "anaemia": anaemia,
        "creatinine_phosphokinase": creatinine_phosphokinase,
        "diabetes": diabetes,
        "ejection_fraction": ejection_fraction,
        "high_blood_pressure": high_blood_pressure,
        "platelets": platelets,
        "serum_creatinine": serum_creatinine,
        "serum_sodium": serum_sodium,
        "sex": sex,
        "smoking": smoking,
        "time": time,
    }
    data = pd.DataFrame([input_data])

    # Lấy dự đoán từ tất cả các mô hình
    predictions = {name: int(model.predict(data)[0]) for name, model in models.items()}

    # Hiển thị kết quả
    st.subheader("Predictions")
    for model_name, prediction in predictions.items():
        result_text = t["survived"] if prediction == 0 else t["death"]
        st.write(f"{model_name}: {result_text}")
