import numpy as np  
import pandas as pd  

# hàm chia node thành 2 node con dựa trên ngưỡng
def split_node(column, threshold_split):  
    left_node = column[column <= threshold_split].index 
    right_node = column[column > threshold_split].index  
    return left_node, right_node 
# hàm tính entropy
def entropy(y_target):  
    values, counts = np.unique(y_target, return_counts = True) 
    result = -np.sum([(count / len(y_target)) * np.log2(count / len(y_target)) for count in counts])
    return result 
# hàm tính information gain
def info_gain(column, target, threshold_split):  
    entropy_start = entropy(target)  

    left_node, right_node = split_node(column, threshold_split)  

    n_target = len(target)  
    n_left = len(left_node)  
    n_right = len(right_node)  

    # tính entropy cho các node con
    entropy_left = entropy(target[left_node])  
    entropy_right = entropy(target[right_node]) 

    # Tính tổng entropy của các node con có trọng số
    weight_entropy = (n_left / n_target) * entropy_left + (n_right / n_target) * entropy_right

    # Tính Information Gain
    ig = entropy_start - weight_entropy
    return ig

# hàm tìm feature và threshold tốt nhất để chia
def best_split(dataX, target, feature_id):  
    best_ig = -1  
    best_feature = None 
    best_threshold = None
    for _id in feature_id:
        column = dataX.iloc[:, _id] 
        thresholds = set(column)
        for threshold in thresholds:  
            ig = info_gain(column, target, threshold) 
            if ig > best_ig: 
                best_ig = ig 
                best_feature = dataX.columns[_id] 
                best_threshold = threshold 
    return best_feature, best_threshold  
# hàm lấy giá trị xuất hiện nhiều nhất trong node lá
def most_value(y_target):  
    value = y_target.value_counts().idxmax()  
    return value 
# lớp Node đại diện cho từng node trong cây
class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None): 
        self.feature = feature  
        self.threshold = threshold  
        self.left = left  
        self.right = right  
        self.value = value  

    def is_leaf_node(self):
        return self.value is not None  

# lớp Decision Tree Classification
class DecisionTreeClass:
    def __init__(self, min_samples_split=2, max_depth=10, n_features=None):
        self.min_samples_split = min_samples_split 
        self.max_depth = max_depth  
        self.root = None  
        self.n_features = n_features  

    def grow_tree(self, X, y, depth=0):  
        n_samples, n_feats = X.shape  
        n_classes = len(np.unique(y))  

        # Điều kiện dừng: nếu đạt độ sâu tối đa hoặc không thể chia thêm
        if (depth >= self.max_depth or n_classes == 1 or n_samples < self.min_samples_split):
            leaf_value = most_value(y)
            return Node(value=leaf_value)  

        # lấy số cột ngẫu nhiên khi tham số n_features khác None
        feature_id = np.random.choice(n_feats, self.n_features, replace=False)
        
        # tìm feature và threshold tốt nhất để chia
        best_feature, best_threshold = best_split(X, y, feature_id)

        # tách node thành node trái và phải
        left_node, right_node = split_node(X[best_feature], best_threshold)

        # dùng đệ quy để xây dựng cây con
        left = self.grow_tree(X.loc[left_node], y.loc[left_node], depth + 1)
        right = self.grow_tree(X.loc[right_node], y.loc[right_node], depth + 1)

        # trả về node hiện tại với thông tin chia và 2 node con
        return Node(best_feature, best_threshold, left, right)

    def fit(self, X, y):  
        # nếu n_features là None, lấy tất cả các feature
        self.n_features = X.shape[1] if self.n_features is None else min(X.shape[1], self.n_features)
        self.root = self.grow_tree(X, y)  

    def traverse_tree(self, x, node):  
        if node.is_leaf_node():
            return node.value

        if x[node.feature] <= node.threshold:
            return self.traverse_tree(x, node.left)
        return self.traverse_tree(x, node.right)
    
    def predict(self, X):  
        return np.array([self.traverse_tree(x, self.root) for index, x in X.iterrows()])


def print_tree(node, indent = ""):
    # nếu node là node lá, in ra giá trị của nó
    if node.is_leaf_node():
        print(f"{indent}Leaf: {node.value}")
        return
    
    # in ra node hiện tại với feature và threshold
    print(f"{indent}Node: If {node.feature} <= {node.threshold:.2f}")

    # in ra cây con bên trái (đúng)
    print(f"{indent}  True:")
    print_tree(node.left, indent + "    ")

    # in ra cây con bên phải (sai)
    print(f"{indent}  False:")
    print_tree(node.right, indent + "    ")
# hàm tính độ chính xác
def accuracy(y_actual, y_pred):  
    acc = np.sum(y_actual == y_pred) / len(y_actual)  
    return acc * 100  

