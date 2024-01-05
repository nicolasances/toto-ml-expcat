from flask import Flask, request
#from model import expcat
from flask_cors import CORS
from config.config import ExpcatConfig
from dlg.infer import infer_category
from dlg.train import train_models

# Load the config
ExpcatConfig()

app = Flask(__name__)
CORS(app, origins=["*"])

@app.route('/', methods=['GET'])
def smoke():
    return {"api": "expcat", "running": True}

@app.route("/train", methods=["POST"])
def train(): 
    return train_models(request)

@app.route('/predict', methods=['GET'])
def predict():
    return infer_category(request)

if __name__ == '__main__':
    app.run()
    
