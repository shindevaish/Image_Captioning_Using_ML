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
    
    if (result.results) {
        resultDiv.innerHTML = '';
        result.results.forEach(imgData => {
            const imgElement = document.createElement('img');
            imgElement.src = '/static/image/' + imgData.file_name; // Adjust path as needed
            imgElement.alt = imgData.caption; // Add alt text for better accessibility

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

document.getElementById('upload-btn').addEventListener('click', function () {
    document.getElementById('file-input').click();
});

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

    // if (result.file_name && result.caption) {
    //     const imgElement = document.createElement('img');
    //     imgElement.src = '/static/uploaded_image/' + result.file_name; // Adjust path as needed
    //     imgElement.alt = result.caption;    // Add alt text for better accessibility

    //     const captionElement = document.createElement('p');
    //     captionElement.innerText = result.caption;

    //     const container = document.createElement('div'); // Create a container for each image and its caption
    //     container.classList.add('img-container');

    //     container.appendChild(imgElement);
    //     container.appendChild(captionElement);
    //     resultDiv.appendChild(container);
    // } else if (result.error) {
    //     resultDiv.innerHTML = `<p>${result.error}</p>`;
    // }

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
});

document.getElementById('add-image-btn').addEventListener('click', function () {
    document.getElementById('file-input').click();
});

document.getElementById('file-input').addEventListener('change', async function () {
    const file = this.files[0];
    const formData = new FormData();
    formData.append('file', file);

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