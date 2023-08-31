import pickle
import sklearn
import nltk
import string
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

punctuation_symbols = list(string.punctuation)

def tokenize(desc, email): 
    '''
    Tokenizes a description and applies the following:
     - removes stop words
     - removes useless characters (e.g. '-')
     - stems the word
    '''
    stemmer = PorterStemmer()

    stopwords_vocab = getStopwords(email)

    # Split descriptions into tokens
    tokens = desc.split()
    
    # Filter out stopwords
    filtered_tokens = [word.lower() for word in tokens if word.lower() not in stopwords_vocab]
    filtered_tokens = [word for word in filtered_tokens if word not in punctuation_symbols]
    
    # Perform stemming
    stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]
    
    # Join the tokens, as needed by TF-IDF
    joined_tokens = " ".join(stemmed_tokens)
    
    return joined_tokens

def loadModel(email): 

    model_name = "expcat-model"
    vectorizer_name = "expcat-vect"

    if email == "carolineboje.m@gmail.com": 
        model_name = "expcat-model-cbm"
        vectorizer_name = "expcat-vect-cbm"

    # Load the vectorizer
    with open(f"model/{vectorizer_name}", "rb") as f:
        vectorizer = pickle.load(f)

    # Load the model
    with open(f"model/{model_name}", "rb") as f:
        model = pickle.load(f)

    return (model, vectorizer)

def getStopwords(email): 

    stopwords_file = "expcat-stopwords"

    if email == "carolineboje.m@gmail.com": 
        stopwords_file = "expcat-stopwords-cbm"

    with open(f"model/{stopwords_file}", "rb") as f: 
        stopwords = pickle.load(f)

    return stopwords

def predict(desc, email): 

    if desc == None: 
        return {}

    # Load the model
    (model, vectorizer) = loadModel(email)

    # Tokenize the description
    tokenized_desc = tokenize(desc, email)

    # Vectorize the description
    X = vectorizer.transform([tokenized_desc])

    pred = model.predict(X)

    return {"prediction": pred.tolist()}