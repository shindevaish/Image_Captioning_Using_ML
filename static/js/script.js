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
    document.getElementById('result').innerText = `Result: ${result.message}`;
});
