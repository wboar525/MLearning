import requests

ADDRESS = 'http://127.0.0.1:8000'

def check_status():
    response = requests.get(f'{ADDRESS}/')
    if response.status_code == 200:
        return response.text

def train_model(model_type):
    response = requests.post(f'{ADDRESS}/train_model/{model_type}')
    if response.status_code == 200:
        return response.text

def model_predict(model_type, input_data):
    response = requests.post(f'{ADDRESS}/predict/{model_type}', json=input_data)
    if response.status_code == 200:
        return response.text

print(check_status())
print(train_model('LogisticRegression'))

test_data = {'age': 50, "who": 'male'}
print(model_predict('LogisticRegression',test_data))
