# Bài 1:
(Chạy file steamlit app1.py)
# Công nghệ sử dụng:
- numpy
- matplotlib
- cvxopt (Quadratic Programming - QP)

# Thuật toán
- Sử dụng SVM với các bước sau đây:
1. Tối ưu hoá để tìm ra giá trị λ cho cực tiểu hàm mục tiêu với việc sử dụng QP lưu ý rằng phải Thỏa mãn các ràng buộc là các λ đều lớn hơn hoặc bằng 0 và tổng của λiyi phải bằng 0.
2. Tính toán trọng số w bằng tổng của các λi nhân với yi và các điểm xi
3. Sau khi xác định được vector hỗ trợ sau đó bắt đầu tính b từ các vector hỗ trợ 
4. Cuối cùng là biểu diễn trực quan đồ thị bằng phương trình w0*x1 + w1*x2 + b ‎ =  0 sau khi chạy plt.show() từ thư viện matplotlib ta thấy biểu đồ các biên dương âm song song cách đều nhau 1 khoảng gọi là margin

# Kết quả
<img width="488" alt="Screenshot 2024-10-18 at 21 51 38" src="https://github.com/user-attachments/assets/1591792a-7874-44d7-b558-0e5d9a2e8fd8">


# Bài 2:
(Chạy file steamlit app2.py)
#Công nghệ sử dụng:
- numpy
- matplotlib
- cvxopt (Quadratic Programming - QP)
# Thuật toán:
- Sử dụng SVM với các bước sau đây:
1. Sử dụng QP từ thư viện cvxopt để tìm các giá trị λ tối ưu
2. weight vector and bias: Khi có giá trị λ mình bắt đầu tính vector trọng số w và b để ra được đường phân tách
3. Tìm các điểm dữ liệu có ảnh hưởng tới đường phân tách được gọi là vector hỗ trợ (support vectors)
4. Sau đó sử dụng plt.show() từ thư viện matplotlib để trực quan hoá dữ liệu, các vector hỗ trợ, và các đường biên dương âm và margin
5. Slack value: tính toán và hiển thị các biến slack(ξ) để đo các điểm nằm ngoài margin

# Kết quả 
<img width="620" alt="Screenshot 2024-10-18 at 21 54 29" src="https://github.com/user-attachments/assets/43c65c5a-3081-4cc4-b3b8-924630586ebc">

# Lưu ý: (giá trị C có thể thay đổi để xem được các đường biên dương âm và margin và giá trị C không được nhỏ hơn 0)