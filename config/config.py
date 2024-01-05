
from typing import List
from totoapicontroller.model.TotoConfig import TotoConfig
from totoapicontroller.model.singleton import singleton

from store.store import LoadedModel, ModelStore

@singleton
class ExpcatConfig(TotoConfig): 
    
    loaded_models: List[LoadedModel]
    
    def __init__(self) -> None:
        super().__init__()
        
        # Load the model (download it), to avoid the donwload time at inference
        self.loaded_models = ModelStore().download_all()
        
        self.logger.log("INIT", "Expcat Configuration loaded.")
        
    def get_api_name(self) -> str:
        return "toto-ml-expcat"
    
    def get_model(self, user: str): 
        
        for model in self.loaded_models: 
            if model.user == user: 
                return model