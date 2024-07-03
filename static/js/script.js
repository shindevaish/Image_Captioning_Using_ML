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
            imgElement.src = image.url; // Assuming the image URL is available in 'url' field
            imgElement.alt = image.caption; // Assuming a caption field is available for the alt text
            resultContainer.appendChild(imgElement);
        });
    } else if (result.error) {
        const errorElement = document.createElement('div');
        errorElement.textContent = `Error: ${result.error}`;
        resultContainer.appendChild(errorElement);
    }

    // Keep the search input value
    document.getElementById('search-input').value = query;
});
