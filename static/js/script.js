document.getElementById('search-btn').addEventListener('click', async function (event) {
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
    const resultDiv = document.getElementById('result');

    if (result.images) {
        resultDiv.innerHTML = '';
        result.images.forEach(imgData => {
            const imgElement = document.createElement('img');
            imgElement.src = imgData.url;
            imgElement.alt = imgData.caption;

            const captionElement = document.createElement('p');
            captionElement.textContent = imgData.caption;

            const imgContainer = document.createElement('div');
            imgContainer.appendChild(imgElement);
            imgContainer.appendChild(captionElement);

            resultDiv.appendChild(imgContainer);
        });
    } else if (result.error) {
        resultDiv.innerHTML = `<p>Error: ${result.error}</p>`;
    }
    
    // Keep the query in the input box
    document.getElementById('search-input').value = query;
});
