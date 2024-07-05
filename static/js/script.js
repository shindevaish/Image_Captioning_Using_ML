document.getElementById('search-form').addEventListener('submit', async function (event) {
    event.preventDefault();
    const query = document.getElementById('search-input').value;
    const response = await fetch('/search/', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query: query }),
    });
    const result = await response.json();
    
    // Clear previous search results
    const resultContainer = document.getElementById('result');
    resultContainer.innerHTML = "";

    // Append new search results
    if (result.images) {
        result.images.forEach(image => {
            const imgElement = document.createElement('img');
            imgElement.src = image.url;
            imgElement.alt = image.caption;
            imgElement.style.width = "200px";  // Example styling
            imgElement.style.margin = "10px"; // Example styling
            resultContainer.appendChild(imgElement);

            const captionElement = document.createElement('div');
            captionElement.textContent = image.caption;
            resultContainer.appendChild(captionElement);
        });
    } else if (result.error) {
        const errorElement = document.createElement('div');
        errorElement.textContent = `Error: ${result.error}`;
        resultContainer.appendChild(errorElement);
    }

    // Keep the search input value
    document.getElementById('search-input').value = query;
});
