
from flask import Request
from totoapicontroller.TotoDelegateDecorator import toto_delegate
from totoapicontroller.model.ExecutionContext import ExecutionContext
from totoapicontroller.model.UserContext import UserContext

from config.config import ExpcatConfig
from model.expcatv2 import ModelExpcat

@toto_delegate(config_class=ExpcatConfig)
def infer_category(req: Request, user_context: UserContext, exec_context: ExecutionContext):
    
    return ModelExpcat(exec_context).infer(user_context.email, req.args.get("description"))


@toto_delegate(config_class=ExpcatConfig)
def bulk_infer_categories(req: Request, user_context: UserContext, exec_context: ExecutionContext):
    
    # Get the descriptions from the body
    descriptions = req.json.get('descriptions')
    
    return ModelExpcat(exec_context).bulk_infer(user_context.email, descriptions)