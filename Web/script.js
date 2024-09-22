document.getElementById('uploadForm').addEventListener('submit', function(event) {
    event.preventDefault(); // Ngăn chặn gửi form mặc định

    const fileInput = document.getElementById('fileInput');
    const imagePreview = document.getElementById('imagePreview');

    if (fileInput.files && fileInput.files[0]) {
        const file = fileInput.files[0];

        // Kiểm tra xem tệp có phải là hình ảnh không
        if (!file.type.startsWith('image/')) {
            alert('Vui lòng chọn một tệp ảnh.');
            return;
        }

        const reader = new FileReader();

        reader.onload = function(e) {
            imagePreview.src = e.target.result;
            imagePreview.style.display = 'block'; // Hiển thị ảnh
        }

        reader.readAsDataURL(file);
    } else {
        alert('Vui lòng chọn một tệp ảnh.');
    }
});
