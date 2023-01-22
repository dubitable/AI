import requests, base64, json

with open("fig.jpg", "rb") as image_file:
    b64 = base64.b64encode(image_file.read())
    b64 = b64.decode("utf-8", "strict")

api = "http://127.0.0.1:8000/figdetector/"
payload = json.dumps({"image": b64})

response = requests.post(api, payload)
body = response.json()

print(body["fruit"], body["value"])