let slideIndex = 0;

function showSlides(n) {
    const slides = document.querySelectorAll('.carousel-item');
    if (n >= slides.length) {
        slideIndex = 0;
    } else if (n < 0) {
        slideIndex = slides.length - 1;
    } else {
        slideIndex = n;
    }

    // Hide all slides
    slides.forEach(slide => slide.classList.remove('active'));

    // Show the current slide
    slides[slideIndex].classList.add('active');
}

function moveSlide(step) {
    showSlides(slideIndex + step);
}

// Initialize the first slide
showSlides(slideIndex);
