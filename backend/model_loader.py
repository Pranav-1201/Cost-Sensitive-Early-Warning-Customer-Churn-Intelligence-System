import pickle
import os

_cache = {}

def load_model():
    if "model" in _cache:
        return _cache["model"], _cache["threshold"]
    
    # Look for model relative to this file's location
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # Try backend/models/ first, then project root models/
    candidates = [
        os.path.join(base_dir, "models", "churn_model.pkl"),
        os.path.join(base_dir, "..", "models", "churn_model.pkl"),
    ]
    
    model_path = None
    for path in candidates:
        if os.path.exists(path):
            model_path = path
            break
    
    if model_path is None:
        raise FileNotFoundError(
            f"churn_model.pkl not found. Tried: {candidates}"
        )
    
    with open(model_path, "rb") as f:
        artifact = pickle.load(f)
    
    # Handle both dict format {"model":..., "threshold":...}
    # and raw model format
    if isinstance(artifact, dict):
        _cache["model"] = artifact["model"]
        _cache["threshold"] = artifact.get("threshold", 0.13)
    else:
        _cache["model"] = artifact
        _cache["threshold"] = 0.13
    
    print(f"✅ Model loaded from {model_path}")
    print(f"✅ Threshold: {_cache['threshold']}")
    print(f"✅ Model type: {type(_cache['model'])}")
    return _cache["model"], _cache["threshold"]