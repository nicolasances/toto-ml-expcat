import json
import os
import re
import tempfile
from google.cloud import storage
from totoapicontroller.model.ExecutionContext import ExecutionContext
from totoapicontroller.TotoLogger import TotoLogger
from totoapicontroller.model.TotoConfig import TotoConfig

class TrainingDataStore: 
    
    def __init__(self, exec_context: ExecutionContext) -> None:
        self.logger = exec_context.logger
        self.cid = exec_context.cid
        

    def load_training_data(self): 
        """Loads the model training data

        Data is loaded in-memory for now, as long as it is not a problem. 
        
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
        training_data = []
        
        with open(local_temp_file.name, "r") as file: 
            
            # Read the lines in the file
            for idx, line in enumerate(file): 
                
                # Read a line and transform to a dict
                expense = json.loads(line)

                # Add the expense to the training data
                training_data.append(expense)
                
                # Print every 500 lines
                if idx > 0 and idx % 1000 == 0: 
                    self.logger.log(self.cid, f"Ingested {idx} expenses")
        
        self.logger.log(self.cid, f"Ingested {idx} expenses")
        
        return training_data
        

if __name__ == '__main__':
    
    class Config(TotoConfig): 
        
        def get_api_name(self): 
            return "test"
        
    exec_context = ExecutionContext(Config(), TotoLogger("test"), "980as8d90as8d0a9s8d")
    
    TrainingDataStore(exec_context).load_training_data()
