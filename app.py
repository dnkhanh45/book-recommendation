from flask import Flask, jsonify, request
import numpy as np

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route("/matrix-factorization", methods=['GET'])
def matrix_factorization():
    if request.method == 'GET':
        user_id = request.args.get('user_id')
        response = jsonify({
            "book_ids": np.load('./recommend/matrix-factorization/{}.npy'.format(user_id)).tolist()
        })
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response

@app.route("/content-based", methods=['GET'])
def content_based():
    if request.method == 'GET':
        user_id = request.args.get('user_id')
        response = jsonify({
            "book_ids": np.load('./recommend/content-based/{}.npy'.format(user_id)).tolist()
        })
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response

if __name__ == '__main__':
    app.run(debug=True, port=5000)