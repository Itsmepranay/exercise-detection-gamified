from pydantic import BaseModel
from datetime import datetime
from app.models.exercise import MetricType

class ExerciseResponse(BaseModel):
    id: int
    name: str
    display_name: str
    description: str | None
    metric_type: MetricType
    created_at: datetime
    
    class Config:
        from_attributes = True


