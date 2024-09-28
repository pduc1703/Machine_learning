document.addEventListener('DOMContentLoaded', function() {
    const images = document.querySelectorAll('.imagePreview');
    images.forEach(image => {
        image.style.display = 'block'; 
    });
});
