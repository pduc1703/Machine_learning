import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split

def loadExcel(filename) -> pd.DataFrame:
    return pd.read_excel(filename)

def splitTrainTest(data, target, ratio=0.25):
    data_X = data.drop([target], axis=1)
    data_y = data[[target]]
    X_train, X_test, y_train, y_test = train_test_split(data_X, data_y, test_size=ratio, random_state=42)
    data_train = pd.concat([X_train, y_train], axis=1)
    return data_train, X_test, y_test

def mean_class(data_train, target):
    df_group = data_train.groupby(by=target).mean()
    return df_group

def target_pred(data_group, data_test):
    dict_ = {}
    for index, value in enumerate(data_group.values):
        result = np.sqrt(np.sum((data_test - value) ** 2, axis=1))
        dict_[index] = result
    df = pd.DataFrame(dict_)
    return df.idxmin(axis=1)

st.title("Euclidean Distance Classifier")
uploaded_file = st.file_uploader("Upload an Excel file", type=["xls", "xlsx"])
if uploaded_file is not None:
    data = loadExcel(uploaded_file)
    st.write("### Data")
    st.write(data)

    target_column = st.selectbox("Select Target Column", data.columns)
    test_ratio = st.slider("Select Test Ratio", 0.1, 0.5, 0.3)
    data_train, X_test, y_test = splitTrainTest(data, target_column, ratio=test_ratio)
    st.write("### Training Data")
    st.write(data_train)

    df_group = mean_class(data_train, target_column)
    st.write("### Class Means")
    st.write(df_group)

   
    predictions = target_pred(df_group, X_test.values)
    df_predictions = pd.DataFrame(predictions, columns=['Predict'])
    
  
    y_test = y_test.reset_index(drop=True)
    y_test.columns = ['Actual']

  
    results = pd.concat([df_predictions, y_test], axis=1)
    st.write("### Predictions vs Actual")
    st.write(results)
else:
    st.warning("Please upload an Excel file to continue.")