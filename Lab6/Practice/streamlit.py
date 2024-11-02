import streamlit as st
import torch
import torch.nn.functional as loss

# Công thức tính CrossEntropy Loss
def crossEntropyLoss(output, target):
    return loss.cross_entropy(output.unsqueeze(0), target.unsqueeze(0))

# Công thức tính Mean Square Error
def meanSquareError(output, target):
    return torch.mean((output - target) ** 2)

# Công thức tính Binary Entropy Loss
def binaryEntropyLoss(output, target, n):
    return loss.binary_cross_entropy(output, target, reduction='sum') / n

# Công thức tính Sigmoid
def sigmoid(x: torch.tensor):
    return 1 / (1 + torch.exp(-x))

# Công thức tính Relu
def relu(x: torch.tensor):
    return torch.max(torch.tensor(0.0), x)

# Công thức tính Soft Max
def softmax(zi: torch.tensor):
    exp_zi = torch.exp(zi)
    return exp_zi / torch.sum(exp_zi)

# Công thức tính Tanh
def tanh(x: torch.tensor):
    return (torch.exp(x) - torch.exp(-x)) / (torch.exp(x) + torch.exp(-x))


st.title("Loss and Activation Function Calculator")
st.subheader("Enter inputs for Loss Functions")

input_values = st.text_input("Input tensor (comma separated values)", "0.1, 0.3, 0.6, 0.7")
target_values = st.text_input("Target tensor (comma separated values)", "0.31, 0.32, 0.8, 0.2")

inputs = torch.tensor([float(x) for x in input_values.split(',')])
target = torch.tensor([float(x) for x in target_values.split(',')])
n = len(inputs)

mse = meanSquareError(inputs, target)
binary_loss = binaryEntropyLoss(inputs, target, n)
cross_loss = crossEntropyLoss(inputs, target)

st.subheader("Loss Function Results")
st.write(f"Mean Square Error: {mse}")
st.write(f"Binary Entropy Loss: {binary_loss}")
st.write(f"Cross Entropy Loss: {cross_loss}")
st.subheader("Enter inputs for Activation Functions")

activation_values = st.text_input("Activation function input tensor (comma separated values)", "1, 5, -4, 3, -2")

x = torch.tensor([float(x) for x in activation_values.split(',')])
f_sigmoid = sigmoid(x)
f_relu = relu(x)
f_softmax = softmax(x)
f_tanh = tanh(x)

st.subheader("Activation Function Results")
st.write(f"Sigmoid: {f_sigmoid}")
st.write(f"ReLU: {f_relu}")
st.write(f"Softmax: {f_softmax}")
st.write(f"Tanh: {f_tanh}")
