from flask import Flask, jsonify, request

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route("/get", methods=['GET'])
def get():
    if request.method == 'GET':
        print(request.args.get('user_id'))
        response = jsonify({
            "book_ids": [1, 2, 3, 4, 5, 6, 7, 8, 9]
        })
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response

if __name__ == '__main__':
    app.run(debug=True)