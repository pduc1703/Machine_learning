#bài 1

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Đọc file dữ liệu (Bạn cần thay đường dẫn tới file của mình)

dataset = "C:\Users\ACER\OneDrive\Tài liệu\GitHub\Machine_learning\Lap2\Education.csv"
data = pd.read_csv(dataset)

# Hiển thị dữ liệu đầu tiên để kiểm tra
print("Dữ liệu đầu tiên:")
print(data.head())

# Sử dụng LabelEncoder để chuyển đổi nhãn từ văn bản thành số
label_encoder = LabelEncoder()
data['Label_2'] = label_encoder.fit_transform(data['Label'])  # Mã hóa cột 'Label' (Positive/Negative)

# In dữ liệu sau khi mã hóa nhãn
print("\nDữ liệu sau khi mã hóa nhãn:")
print(data[['Text', 'Label', 'Label_2']].head())

# Sử dụng CountVectorizer để chuyển đổi văn bản thành ma trận đặc trưng
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data['Text'])  # Chuyển đổi cột 'Text' thành ma trận số

# Cột 'Label_2' là nhãn đã mã hóa (0: Negative, 1: Positive)
y = data['Label_2']

# Chia dữ liệu thành tập huấn luyện và tập kiểm thử
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Áp dụng Bernoulli Naive Bayes
bernoulli_nb = BernoulliNB()  # Khởi tạo mô hình Bernoulli Naive Bayes
bernoulli_nb.fit(X_train, y_train)  # Huấn luyện mô hình
y_pred_bernoulli = bernoulli_nb.predict(X_test)  # Dự đoán trên tập kiểm thử

# Đánh giá kết quả Bernoulli Naive Bayes
print("\nBernoulli Naive Bayes Results:")
print("Accuracy:", accuracy_score(y_test, y_pred_bernoulli))
print("Classification Report:")
print(classification_report(y_test, y_pred_bernoulli))

# Áp dụng Multinomial Naive Bayes
multinomial_nb = MultinomialNB()  # Khởi tạo mô hình Multinomial Naive Bayes
multinomial_nb.fit(X_train, y_train)  # Huấn luyện mô hình
y_pred_multinomial = multinomial_nb.predict(X_test)  # Dự đoán trên tập kiểm thử

# Đánh giá kết quả Multinomial Naive Bayes
print("\nMultinomial Naive Bayes Results:")
print("Accuracy:", accuracy_score(y_test, y_pred_multinomial))
print("Classification Report:")
print(classification_report(y_test, y_pred_multinomial))

# So sánh tổng quát giữa hai mô hình
print("\nSo sánh kết quả giữa Bernoulli và Multinomial Naive Bayes:")
print(f"Bernoulli Naive Bayes Accuracy: {accuracy_score(y_test, y_pred_bernoulli)}")
print(f"Multinomial Naive Bayes Accuracy: {accuracy_score(y_test, y_pred_multinomial)}")

