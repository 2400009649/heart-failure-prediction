import streamlit as st
import sys
import sklearn

# Thông tin phiên bản Python và scikit-learn
st.sidebar.title("Environment Information")
st.sidebar.write(f"Python version: {sys.version}")
st.sidebar.write(f"scikit-learn version: {sklearn.__version__}")
