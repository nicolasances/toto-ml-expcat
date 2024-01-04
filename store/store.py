import os
from google.cloud import storage

class TrainingDataStore: 

    def load_training_data(): 
        """Loads the model training data

        Data is loaded in-memory for now, as long as it is not a problem. 
        """
        # Get access to the GCS Bucket
        client = storage.client()
        bucket = client.bucket(os.environ["BACKUP_BUCKET"])

        # Find the latest Expenses Backup file
        files = bucket.list_blobs()

        for file in files: 
            # Exclude folders
            if file.name.endswith("/"):
                continue

            print(file.name)


if __name__ == '__main__':
    TrainingDataStore().load_training_data()
