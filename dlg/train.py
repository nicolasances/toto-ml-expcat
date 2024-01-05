
from flask import Request
from totoapicontroller.TotoDelegateDecorator import toto_delegate
from totoapicontroller.model.ExecutionContext import ExecutionContext
from totoapicontroller.model.UserContext import UserContext

from config.config import ExpcatConfig
from model.expcatv2 import ModelExpcat

@toto_delegate(config_class=ExpcatConfig)
def train_models(req: Request, user_context: UserContext, exec_context: ExecutionContext):
    
    return ModelExpcat(exec_context).train()