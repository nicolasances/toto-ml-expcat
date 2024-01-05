from json import load
import string
from sklearn.neural_network import MLPClassifier
from config.config import ExpcatConfig
from model.preprocessing.tokenize import tokenize_description
from model.preprocessing.vectorization import create_vocab, custom_encode
from store.store import ModelStore, TrainingDataStore
from totoapicontroller.model.ExecutionContext import ExecutionContext
from nltk.stem import PorterStemmer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class ModelExpcat: 
    
    def __init__(self, exec_context: ExecutionContext) -> None:
        self.exec_context = exec_context
        self.logger = exec_context.logger
        self.cid = exec_context.cid
        self.punctuation_symbols = list(string.punctuation)
        self.stemmer = PorterStemmer()
        self.model_store = ModelStore()
        self.config: ExpcatConfig = exec_context.config
        
    def infer(self, user: str, desc: str):
        """Infers the category of an expense given its description

        Args:
            user (str): the user email
            desc (str): the description of the expense

        Returns:
            dict: a dictionnary with one entry: the predicted category of the expense
        """
        self.logger.log(self.cid, f"Inference started: user [{user}], expense description [{desc}]")
        
        # Load the vocabulary
        loaded_model = self.config.get_model(user)
        model = loaded_model.model
        vocab = loaded_model.vocab
        
        # Feature Engineering
        tokens = tokenize_description(desc, self.punctuation_symbols, self.stemmer).split()
        embedding = custom_encode(tokens, vocab)
        
        # Infer
        prediction = model.predict([embedding])
        
        self.logger.log(self.cid, f"Prediction: {prediction}")
        
        return {"category": prediction[0]}
    
    def bulk_infer(self, user: str, descriptions: list): 
        """Bulk inference of the category of multiple expenses. 

        Args:
            user (str): the user email
            descriptions (list): a list of the description of each expense for which the category needs to be predicted

        Returns:
            dict: a dict with one entry: "categories", which value is a list of predicted categories (str)
        """
        self.logger.log(self.cid, f"Batch Inference started: user [{user}], number of expenses to categorize: [{len(descriptions)}]")
        
        # Load the vocabulary
        loaded_model = self.config.get_model(user)
        model = loaded_model.model
        vocab = loaded_model.vocab
        
        # Feature Engineering
        tokens = [tokenize_description(desc, self.punctuation_symbols, self.stemmer).split() for desc in descriptions]
        embeddings = [custom_encode(t, vocab) for t in tokens]
        
        # Infer 
        prediction = model.predict(embeddings)
        
        self.logger.log(self.cid, f"Categories predicted.")
        
        return {"categories": prediction.tolist()}
    

    def train(self):
        """Trains the model on the latest data

        The training process performs the following: 
        - Download data from the last backup
        - Splits the data in training and test
        - Trains 1 model for each user
        - Scores the model on both training and test data
        - Persists the trained model (as the latest version) and metrics 
        """
        self.logger.log(self.cid, f"Training of ExpCat starting")
        
        # 1. Download data from the latest backup
        training_data, training_filename = TrainingDataStore(self.exec_context).load_training_data()
        
        self.logger.log(self.cid, f"Training data downloaded. Using file [{training_filename}]")
        
        # 2. Train a Model for each user
        accuracies = {}
        
        for user, data in training_data.items(): 
            
            self.logger.log(self.cid, f"Training model for user [{user}]")
            
            # Create a dataframe with the data and select the base features
            user_data = pd.DataFrame(data)[["description", "category"]]
            
            # Feature Engineering 
            # Tokenize the description
            user_data.loc[:,"tokens"] = user_data["description"].apply(tokenize_description, args=(self.punctuation_symbols, self.stemmer))
            # Split tokens
            user_data["split_tokens"] = user_data["tokens"].apply(str.split)
            # Create the vocabulary
            vocab = create_vocab(user_data["split_tokens"].values)
            # Encode the tokens
            user_data["encoded_tokens"] = user_data["split_tokens"].apply(lambda x: custom_encode(x, vocab))
            
            # Train-test split
            train_df, test_df = train_test_split(user_data, test_size=0.2)
            
            # Creates X_train and Y_train
            X_train = train_df["encoded_tokens"].values.tolist()
            Y_train = train_df["category"].values.tolist()
            
            X_test = test_df["encoded_tokens"].values.tolist()
            Y_test = test_df["category"].values.tolist()
            
            self.logger.log(self.cid, f"Training set size: [{len(X_train)}]. Test set size: [{len(X_test)}]")
            
            # Train the model
            model = MLPClassifier(hidden_layer_sizes=(50, 25), max_iter=500, random_state=23, alpha=0.1)
            
            self.logger.log(self.cid, f"Fitting [{model}]")
            
            model.fit(X_train, Y_train)
            
            self.logger.log(self.cid, f"Fitting [{model}] completed")
            
            # Evaluate the model
            Y_test_pred = model.predict(X_test)
            Y_train_pred = model.predict(X_train)
            
            accuracy = accuracy_score(Y_test, Y_test_pred)
            train_accuracy = accuracy_score(Y_train, Y_train_pred)
            
            self.logger.log(self.cid, f"Model Training score: {train_accuracy}")
            self.logger.log(self.cid, f"Model Test score: {accuracy}")
            
            # Save the model and vocabulary
            model_filename = self.model_store.save_model(user, model)
            vocabulary_filename = self.model_store.save_vocabulary(user, vocab)
            
            self.logger.log(self.cid, f"Model saved under [{model_filename}]. Vocabulary: [{vocabulary_filename}]")
            
            # Save the accuracies
            accuracies[user] = {
                "training_accuracy": train_accuracy, 
                "test_accuracy": accuracy
            }
        
        # Reload all the models
        self.config.reload_models()
        
        # Return the data
        return {
            "model": "expcat", 
            "accuracy": accuracies
        }
            
            
            