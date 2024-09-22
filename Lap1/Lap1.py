import numpy as np
import matplotlib.pyplot as plt
#Cho chiều cao của các bạn sinh viên trong lớp lần lượt là (180, 162, 183, 174, 160, 163, 180, 165, 175, 170, 170, 169, 168, 175, 169, 171, 155, 158, 175, 165)
X = np.array([180, 162, 183, 174, 160, 163, 180, 165, 175, 170, 170, 169,168, 175, 169, 171, 155, 158, 175, 165]).reshape(-1,1)
# Cho cân nặng của các bạn sinh viên trong lớp lần lượt là( 86, 55, 86.5, 70, 62, 54, 60, 72, 93, 89, 60, 82, 59, 75, 56, 89, 45, 60, 60, 72)
y = np.array([86, 55, 86.5, 70, 62, 54, 60, 72, 93, 89, 60, 82, 59, 75, 56, 89, 45, 60, 60, 72]).reshape((-1,1))
# Chuyển Y về dạng ma trận 1 cột và 20 dòng
X = np.insert(X, 0, 1, axis=1)

def calculate_theta(X, y):
    # Tính chuyển vị của X
    X_transpose = np.transpose(X)
    # Tính ma trận (X^T * X)^-1
    inverse = np.linalg.inv(np.dot(X_transpose, X))
    # Tính theta
    theta = np.dot(np.dot(inverse, X_transpose), y)
    return theta

def predict(theta0, theta1, x):
    y_pred = theta0 + theta1 * x
    return y_pred

theta = calculate_theta(X, y)
x1 = 150
y1 = theta[0] + theta[1] * x1
x2 = 190
y2 = theta[0] + theta[1] * x2
plt.plot([x1, x2], [y1, y2], 'r-')
plt.plot(X[:,1], y[:,0], 'bo')
plt.xlabel('Chiều cao')
plt.ylabel('Cân nặng')
plt.title('Chiều cao và cân nặng của sinh viên VLU')
plt.show()