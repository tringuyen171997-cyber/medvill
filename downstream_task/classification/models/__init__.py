# models/__init__.py
from models.model import MultimodalBertClf  # ← import the full classifier

def get_model(args):
    return MultimodalBertClf(args)          # ← return full model, not just image encoder