document.getElementById('loginForm').addEventListener('submit', function(event) {
    event.preventDefault();

    var username = document.getElementById('username').value;
    var password = document.getElementById('password').value;

    // Check if fields are not empty
    if (!username || !password) {
        alert('Please enter both username and password');
        return;
    }

    // Send login request to server
    fetch('/validate_login', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            'username': username,
            'password': password
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            // Store user info and redirect
            sessionStorage.setItem('username', username);
            window.location.href = '/main_menu';
        } else {
            alert('Invalid username or password');
        }
    })
    .catch(error => {
        console.error('Error:', error);
        alert('An error occurred during login');
    });
});




