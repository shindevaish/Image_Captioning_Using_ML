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
            imgElement.src = '/static/image/' + imgData.file_name; // Adjust path as needed
            imgElement.alt = imgData.caption;    // Add alt text for better accessibility

            const captionElement = document.createElement('p');
            captionElement.innerText = imgData.caption;

            const container = document.createElement('div'); // Create a container for each image and its caption
            container.classList.add('img-container');

            container.appendChild(imgElement);
            container.appendChild(captionElement);
            resultDiv.appendChild(container);

        });
    } else if (result.detail) {
        resultDiv.innerHTML = `<p>${result.detail}</p>`;
    }
    
    // Keep the query in the input box
    document.getElementById('search-input').value = query;
});
