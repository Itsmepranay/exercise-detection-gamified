import requests

# Test health endpoint
try:
    response = requests.get('http://127.0.0.1:8001/health')
    print(f"Health check: {response.status_code}")
except Exception as e:
    print(f"Health check failed: {e}")

# Test video upload endpoint with a dummy file
try:
    with open('test_video.mp4', 'wb') as f:
        f.write(b'dummy video content')
    files = {'video': open('test_video.mp4', 'rb')}
    data = {'exercise_type': 'plank', 'user_id': 1}
    response = requests.post('http://127.0.0.1:8001/sessions/upload', files=files, data=data)
    print(f"Video upload: {response.status_code}")
    print(f"Response: {response.text}")
except Exception as e:
    print(f"Video upload failed: {e}")