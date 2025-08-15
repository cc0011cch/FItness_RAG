import requests
import json
question = 'Is the Lat Pulldown considered a strength training activity, and if so, why?'

print('question: {}'.format(question))
data={'question': question}

url= 'http://localhost:5000/question'

response = requests.post(url, json=data)
data=json.loads(response.text)
print(response.status_code)
print('question: {}'.format(data['reply']['queston']))
print('answer: {}'.format(data['reply']['answer']))
print('think_logic: {}'.format(data['reply']['think_logic']))

