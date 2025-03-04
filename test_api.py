import requests

url = "http://localhost:8000/ocr/"
file = {"file": open("sudoku.jpg", "rb")}

response = requests.post(url, files=file)
print(response.json())
