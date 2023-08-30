from flask import Flask, jsonify, request
from model import expcat
from flask_cors import CORS

app = Flask(__name__)
CORS(app, origins=["http://localhost:3000", "https://toto-reactjs-expenses-6lv62poq7a-ew.a.run.app"])

@app.route('/predict', methods=['GET'])
def predict():
    return expcat.predict(request.args.get("description"))

if __name__ == '__main__':
    app.run()
