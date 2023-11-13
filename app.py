from flask import Flask, jsonify, request
from model import expcat
from flask_cors import CORS

app = Flask(__name__)
CORS(app, origins=["http://localhost:3000", "https://toto-reactjs-expenses-6lv62poq7a-ew.a.run.app", "https://toto-reactjs-expenses-zkz7cuplpa-ew.a.run.app"])

@app.route('/', methods=['GET'])
def smoke():
    print("GET /")
    return {"api": "expcat", "running": True}

@app.route('/predict', methods=['GET'])
def predict():
    print("GET /predict")
    return expcat.predict(request.args.get("description"), request.args.get("email"))

if __name__ == '__main__':
    app.run()
