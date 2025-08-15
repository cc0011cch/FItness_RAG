from flask import Flask, request, jsonify
from rag import RAG

app = Flask(__name__)
rag = RAG()
@app.route('/question', methods = ['POST'])
def handle_question():
    data = request.json
    question = data['question']
    answer_data = rag.answer(question)
    result={'question': question,
            'reply': answer_data
            }
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug = True)