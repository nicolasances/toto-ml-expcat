import time 
import json
import os
import pickle
import re
import tempfile
from typing import List
from google.cloud import storage
from totoapicontroller.model.ExecutionContext import ExecutionContext
from totoapicontroller.TotoLogger import TotoLogger
from totoapicontroller.model.TotoConfig import TotoConfig

class TrainingDataStore: 
    
    def __init__(self, exec_context: ExecutionContext) -> None:
        self.logger = exec_context.logger
        self.cid = exec_context.cid
        

    def load_training_data(self) -> (dict, str): 
        """Loads the training data from Google Cloud Storage

        Returns:
            dict: a dictionnary where the key is the user email and the value is a list containing the training data 
            str: the name of the downloaded file where the training data was stored
        """
        # Get access to the GCS Bucket
        client = storage.Client()
        bucket = client.bucket(os.environ["BACKUP_BUCKET"])

        # Find the latest Expenses Backup file
        files = bucket.list_blobs()

        latest_date = 0

        for file in files: 
            
            # If the filenae fits the pattern (date-expenses.json)
            if re.match(r"\d{8}-expenses\.json$", file.name): 
                
                # Extract the date 
                file_date = int(file.name[0:8])
                
                # Check if it's the latest currently seen
                if file_date >= latest_date: 
                    latest_date = file_date

        latest_filename = f"{latest_date}-expenses.json"

        latest_file = bucket.blob(latest_filename)
        
        # Download the file
        local_temp_file = tempfile.NamedTemporaryFile()
        latest_file.download_to_filename(local_temp_file.name)
        
        # Open the file and parse the data
        training_data = {}
        
        with open(local_temp_file.name, "r") as file: 
            
            # Read the lines in the file
            for idx, line in enumerate(file): 
                
                # Read a line and transform to a dict
                expense = json.loads(line)
                
                # Get the user
                user = expense["user"]

                # Add the expense to the training data
                if user not in training_data: 
                    training_data[user] = []
                    
                training_data[user].append(expense)
                
                # Print every 500 lines
                if idx > 0 and idx % 1000 == 0: 
                    self.logger.log(self.cid, f"Ingested {idx} expenses")
        
        self.logger.log(self.cid, f"Ingested {idx} expenses")
        
        return training_data, latest_filename
    
class LoadedModel: 
    
    user: str
    model: any
    vocab: dict
    
    def __init__(self, user: str, model: any, vocab: dict) -> None:
        self.user = user
        self.model = model
        self.vocab = vocab

class ModelStore: 
    
    def __init__(self) -> None:
        self.client = storage.Client()
        self.bucket = self.client.bucket(os.environ["MODELS_BUCKET"])
        
    
    def save_model(self, user: str, model: any):
        
        # Define the filepath
        filepath = f"expcat/expcat-{user}"
        
        # Locally dump the model
        local_temp_file = tempfile.NamedTemporaryFile()
        
        with open(local_temp_file.name, "wb") as f: 
            pickle.dump(model, f)
            
        # Upload to the bucket
        blob = self.bucket.blob(filepath)
        
        blob.upload_from_file(local_temp_file)
        
        return filepath
    
    
    def save_vocabulary(self, user: str, vocab: any): 
        """Saves the vocabulary to file

        Args:
            user (str): the email of the user
            vocab (dict): the vocabulary as a dict

        Returns:
            str: the filepath where the vocabulary was saved, in the GCS bucket
        """
        
        # Define the filepath
        filepath = f"expcat/vocab-{user}"
        
        # Locally dump the model
        local_temp_file = tempfile.NamedTemporaryFile()
        
        with open(local_temp_file.name, "wb") as f: 
            pickle.dump(vocab, f)
            
        # Upload to the bucket
        blob = self.bucket.blob(filepath)
        
        blob.upload_from_file(local_temp_file)
        
        return filepath
    
    def get_users_list(self) -> list:
        """Retrieves the list of users that have a trained model saved on GCS

        Returns:
            list: the list of str (users' emails)
        """
        # File prefix, to use to only select the model files for the expcat model
        prefix = "expcat/expcat-"
        
        # List the files in the bucket
        blobs = self.bucket.list_blobs(prefix=prefix)
        
        # For each file, extract the filename and the the user
        users = [blob.name[len(prefix):] for blob in blobs]
        
        return users
    
    def download_all(self) -> List[LoadedModel]:
        """Downloads all model and vocab files for all users

        Returns:
            list: a list of LoadedModel
        """
        # Gets the list of users
        users = self.get_users_list()
        
        # For each user, download the model
        loaded_models = []
        
        for user in users: 
            model, vocab, _ = self.load_model_and_vocab(user)
            loaded_models.append(LoadedModel(user, model, vocab))
        
        return loaded_models
    
    def load_model_and_vocab(self, user: str): 
        """Loads in-memory the Expcat model and vocabulary for the user

        Args:
            user (str): the user (email)

        Returns:
            (model, dict, dict): a tuple containing the loaded model, the vocabulary (as a dict) and a dictionnary that contains the download and load times
        """
        # Define files to download
        model_filepath = f"expcat/expcat-{user}"
        vocab_filepath = f"expcat/vocab-{user}"
        
        # Download the files
        download_start_time = time.time()
        
        model_blob = self.bucket.blob(model_filepath)
        vocab_blob = self.bucket.blob(vocab_filepath)
        
        target_model_file = tempfile.NamedTemporaryFile()
        target_vocab_file = tempfile.NamedTemporaryFile()
        
        model_blob.download_to_filename(target_model_file.name)
        vocab_blob.download_to_filename(target_vocab_file.name)
        
        download_time = time.time() - download_start_time
        
        # Load the model
        load_start_time = time.time()
        
        with open(target_model_file.name, "rb") as f: 
            model = pickle.load(f)
        
        with open(target_vocab_file.name, "rb") as f: 
            vocab = pickle.load(f)
            
        load_time = time.time() - load_start_time
        
        return model, vocab, {"download_time": download_time, "load_time": load_time}

if __name__ == '__main__':
    
    class Config(TotoConfig): 
        
        def get_api_name(self): 
            return "test"
        
    exec_context = ExecutionContext(Config(), TotoLogger("test"), "980as8d90as8d0a9s8d")
    
    training_data = TrainingDataStore(exec_context).load_training_data()
    
    print(training_data)
