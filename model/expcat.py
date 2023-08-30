import pickle
import sklearn

def predict(desc): 

    if desc == None: 
        return {}

    with open("model/expcat-vect", "rb") as f:
        vectorizer = pickle.load(f)

    with open("model/expcat-model", "rb") as f:
        model = pickle.load(f)

    X = vectorizer.transform([desc])

    pred = model.predict(X)

    return {"prediction": pred.tolist()}