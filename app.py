from flask import Flask, jsonify, request
from model import expcat
from flask_cors import CORS

app = Flask(__name__)
CORS(app, origins=["*"])

@app.route('/', methods=['GET'])
def smoke():
    return {"api": "expcat", "running": True}

# @app.route('/predict', methods=['GET'])
# def predict():
#     return expcat.predict(request.args.get("description"), request.args.get("email"))

if __name__ == '__main__':
    app.run()
