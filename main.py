from flask import Flask, jsonify, request
from model import expcat

app = Flask(__name__)

@app.route('/predict', methods=['GET'])
def predict():
    return expcat.predict(request.args.get("description"))

if __name__ == '__main__':
    app.run()
