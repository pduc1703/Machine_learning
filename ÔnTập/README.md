# Câu 4: KNN nhận xét về kể quả
1. Kernel Linear:
- Đơn giản và nhanh nhất trong huấn luyện
- Phù hợp với dữ liệu có thể tách biệt tuyến tính
- Thường cho độ chính xác khá tốt trên bộ dữ liệu Digits

2. Kernel RBF (Radial Basis Function):
- Thường cho độ chính xác cao nhất
- Thời gian huấn luyện lâu hơn kernel tuyến tính
- Có khả năng tổng quát hóa tốt

3. Kernel Polynomial:
- Độ chính xác thường nằm giữa linear và RBF
- Thời gian huấn luyện có thể khá lâu
- Dễ bị overfitting nếu bậc đa thức cao


# Kết luận: Đối với bộ dữ liệu Digits, ta nên chọn mô hình tùy theo nhu cầu
- Nếu cần độ chính xác cao nhất: sử dụng kernel RBF
- Nếu cần tốc độ huấn luyện nhanh: sử dụng kernel linear
- Kernel polynomial ít khi là lựa chọn tốt nhất cho bài toán này
