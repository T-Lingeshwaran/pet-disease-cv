const API_BASE_URL = "https://pet-disease-cv.onrender.com";

// Automatically configure the backend URL.
document.addEventListener("DOMContentLoaded", () => {
    const backendInput = document.querySelector(
        'input[type="url"], input[name="backendUrl"], #backendUrl, #backend-url'
    );

    if (backendInput) {
        backendInput.value = API_BASE_URL;
    }
});