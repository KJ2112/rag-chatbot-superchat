 # Sample Python Documentation

 ## Making HTTP Requests

 The requests library makes it easy:

 ```python
 import requests
 response = requests.get('https://api.example.com/data')
 data = response.json()
 ```

 ## Error Handling

 ```python
 try:
     response = requests.get(url, timeout=5)
     response.raise_for_status()
 except requests.exceptions.RequestException as e:
     print(f"Error: {e}")
 ```

