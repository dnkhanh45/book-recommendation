from flask import Flask, jsonify, request
import numpy as np

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route("/recommend", methods=['GET'])
def recommend():
    if request.method == 'GET':
        user_id = request.args.get('user_id')
        response = jsonify({
            "book_ids": np.load('./recommend/{}.npy'.format(user_id)).tolist()
        })
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response

if __name__ == '__main__':
    app.run(debug=True)