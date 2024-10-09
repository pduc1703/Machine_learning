import streamlit as st
import numpy as np
import pandas as pd

def loadCsv(filename) -> pd.DataFrame:
    df = pd.read_csv(filename)
    return df


def splitTrainTest(data, ratio_test):
    np.random.seed(28)
    index_permu = np.random.permutation(len(data))
    data_permu = data.iloc[index_permu]
    len_test = int(len(data_permu) * ratio_test)
    test_set = data_permu.iloc[: len_test, :]
    train_set = data_permu.iloc[len_test:, :]
    X_train = train_set.iloc[:, :-1]
    y_train = train_set.iloc[:, -1]
    X_test = test_set.iloc[:, :-1]
    y_test = test_set.iloc[:, -1]
    return X_train, y_train, X_test, y_test

def get_words_frequency(data_X):
    bag_words = np.concatenate([i[0].split(' ') for i in data_X.values], axis=None)
    bag_words = np.unique(bag_words)
    matrix_freq = np.zeros((len(data_X), len(bag_words)), dtype=int)
    word_freq = pd.DataFrame(matrix_freq, columns=bag_words)
    for id, text in enumerate(data_X.values.reshape(-1)):
        for j in bag_words:
            word_freq.at[id, j] = text.split(' ').count(j)
    return word_freq, bag_words


def transform(data_test, bags):
    matrix_0 = np.zeros((len(data_test), len(bags)), dtype=int)
    frame_0 = pd.DataFrame(matrix_0, columns=bags)
    for id, text in enumerate(data_test.values.reshape(-1)):
        for j in bags:
            frame_0.at[id, j] = text.split(' ').count(j)
    return frame_0


def cosine_distance(train_X_number_arr, test_X_number_arr):
    dict_kq = {}
    for id, arr_test in enumerate(test_X_number_arr, start=1):
        q_i = np.sqrt(sum(arr_test ** 2))
        for j in train_X_number_arr:
            _tu = sum(j * arr_test)
            d_j = np.sqrt(sum(j ** 2))
            _mau = d_j * q_i
            kq = _tu / _mau
            if id in dict_kq:
                dict_kq[id].append(kq)
            else:
                dict_kq[id] = [kq]
    return dict_kq

class KNNText:
    def __init__(self, k):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        _distance = cosine_distance(self.X_train, X_test)
        self.y_train.index = range(len(self.y_train))
        _distance_frame = pd.concat([pd.DataFrame(_distance), pd.DataFrame(self.y_train)], axis=1)
        target_predict = {}
        for i in range(1, len(X_test) + 1):
            sorted_distances = _distance_frame[[i, 'target']].sort_values(by=i, ascending=True).head(self.k)
            most_common_target = sorted_distances['target'].mode()[0] if not sorted_distances['target'].empty else None
            target_predict[i] = most_common_target
        return target_predict

    def score(self, X_test, y_test):
        predictions = self.predict(X_test)
        correct_predictions = sum(predictions[i] == y_test.iloc[i - 1] for i in predictions)
        accuracy = correct_predictions / len(y_test)
        return accuracy

st.title("KNN Text Classifier with Cosine Distance")
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
if uploaded_file is not None:
    data = loadCsv(uploaded_file)
    st.write("### Data Preview")
    st.write(data.head())
    
    data['Text'] = data['Text'].str.replace(',', '').str.replace('.', '')
    
    test_ratio = st.slider("Select Test Ratio", 0.1, 0.5, 0.25)
    k_value = st.slider("Select k value for KNN", 1, 10, 3)
    
    X_train, y_train, X_test, y_test = splitTrainTest(data, test_ratio)
    words_train_fre, bags = get_words_frequency(X_train)
    words_test_fre = transform(X_test, bags)
    
    knn = KNNText(k=k_value)
    knn.fit(words_train_fre.values, y_train)
    predictions = knn.predict(words_test_fre.values)
    
    pred_df = pd.DataFrame(list(predictions.values()), columns=['Predict'])
    y_test = y_test.reset_index(drop=True)
    y_test = y_test.to_frame(name='Actual')
    result = pd.concat([pred_df, y_test], axis=1)
    
    st.write("### Predictions vs Actual")
    st.write(result)
    
    accuracy = knn.score(words_test_fre.values, y_test['Actual'])
    st.write(f"### Accuracy: {accuracy:.2%}")


    csv = result.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Result as CSV",
        data=csv,
        file_name='KNN_Predictions.csv',
        mime='text/csv'
    )
else:
    st.warning("Please upload a CSV file to continue.")