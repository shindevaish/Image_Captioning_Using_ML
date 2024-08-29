// Event listener for the search button
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

    // Clear previous results
    resultDiv.innerHTML = '';

    if (result.results && result.results.length > 0) {
        result.results.forEach(imgData => {
            const imgElement = document.createElement('img');
            imgElement.src = '/static/image/' + imgData.Title; // Use image_id (Title) to set image source path
            imgElement.alt = imgData.Abstract;

            const captionElement = document.createElement('p');
            captionElement.innerText = `Caption: ${imgData.Abstract}`;

            const container = document.createElement('div');
            container.classList.add('img-container');

            container.appendChild(imgElement);
            container.appendChild(captionElement);
            resultDiv.appendChild(container);
        });
    } else if (result.detail) {
        resultDiv.innerHTML = `<p>${result.detail}</p>`;
    } else {
        resultDiv.innerHTML = `<p>No results found.</p>`;
    }

    // Keep the query in the input box
    document.getElementById('search-input').value = query;
});

// Event listener for the upload button
document.getElementById('upload-btn').addEventListener('click', function () {
    document.getElementById('file-input').click();
});

// Event listener for file input change (upload)
document.getElementById('file-input').addEventListener('change', async function () {
    const file = this.files[0];
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch('/upload/', {
        method: 'POST',
        body: formData,
    });

    const result = await response.json();
    const resultDiv = document.getElementById('result');

    // Clear previous results
    resultDiv.innerHTML = '';

    if (result.images && result.images.length > 0) {
        result.images.forEach(imgData => {
            const imgElement = document.createElement('img');
            imgElement.src = '/static/image/' + imgData.file_name; // Use file_name to set image source path
            imgElement.alt = imgData.caption;

            const captionElement = document.createElement('p');
            captionElement.innerText = `Caption: ${imgData.caption}`;

            const container = document.createElement('div');
            container.classList.add('img-container');

            container.appendChild(imgElement);
            container.appendChild(captionElement);
            resultDiv.appendChild(container);
        });
    } else if (result.error) {
        resultDiv.innerHTML = `<p>${result.error}</p>`;
    }
});

// Event listener for the add-image button
document.getElementById('add-image-btn').addEventListener('click', function () {
    document.getElementById('file-input').click();
});

// Event listener for file input change (add image)
document.getElementById('file-input').addEventListener('change', async function () {
    const file = this.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);

    // Send the file to the add_image API endpoint
    const response = await fetch('/add_image/', {
        method: 'POST',
        body: formData,
    });

    const result = await response.json();
    const resultDiv = document.getElementById('result');

    if (result.file_name && result.caption) {
        resultDiv.innerHTML = `<p>Image uploaded successfully with ID: ${result.file_name} and caption: ${result.caption}</p>`;
    } else if (result.error) {
        resultDiv.innerHTML = `<p>${result.error}</p>`;
    }
});
