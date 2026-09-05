from pydantic import BaseModel, Field

class TransactionRequest(BaseModel):
    step: int = Field(..., ge=1, le=744, example=180)
    amount: float = Field(..., ge=0.0, example=500000.0)
    oldbalanceOrg: float = Field(..., ge=0.0, example=500000.0)
    newbalanceOrig: float = Field(..., ge=0.0, example=0.0)
    oldbalanceDest: float = Field(..., ge=0.0, example=0.0)
    newbalanceDest: float = Field(..., ge=0.0, example=0.0)
    is_transfer: str = Field(..., example="TRANSFER")

class AssessmentResponse(BaseModel):
    risk_score: float
    decision: str
    primary_driver: str
    audit_summary: str
    feature_vector: dict