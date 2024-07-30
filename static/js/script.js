// static/script.js
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
            resultDiv.appendChild(imgElement);
        });
    } else if (result.detail) {
        resultDiv.innerHTML = `<p>${result.detail}</p>`;
    }
    
    // Keep the query in the input box
    document.getElementById('search-input').value = query;
});
