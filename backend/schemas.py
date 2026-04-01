from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

class CustomerInput(BaseModel):
    gender: str = "Male"
    SeniorCitizen: int = 0
    Partner: str = "No"
    Dependents: str = "No"
    tenure: int = Field(default=1, ge=0, le=72)
    PhoneService: str = "Yes"
    MultipleLines: str = "No"
    InternetService: str = "Fiber optic"
    OnlineSecurity: str = "No"
    OnlineBackup: str = "No"
    DeviceProtection: str = "No"
    TechSupport: str = "No"
    StreamingTV: str = "No"
    StreamingMovies: str = "No"
    Contract: str = "Month-to-month"
    PaperlessBilling: str = "Yes"
    PaymentMethod: str = "Electronic check"
    MonthlyCharges: float = Field(default=70.0, ge=0)
    TotalCharges: float = Field(default=150.0, ge=0)

class PredictionResponse(BaseModel):
    probability: float
    prediction: int
    risk_level: str
    threshold_used: float
    shap_values: Optional[Dict[str, float]] = None

class ModelMetric(BaseModel):
    name: str
    accuracy: float
    roc_auc: float
    pr_auc: Optional[float]
    cost: Optional[float]
    status: str
    confusion_matrix: List[List[int]]
    threshold: Optional[float]
    cv_scores: List[float]

class MetricsResponse(BaseModel):
    models: List[ModelMetric]
    best_model: str
    best_threshold: float
    cost_fn: int
    cost_fp: int