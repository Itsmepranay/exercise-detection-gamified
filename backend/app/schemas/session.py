from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional

class ExerciseSessionCreate(BaseModel):
    user_id: int
    exercise_id: int
    score: int = Field(..., ge=0)
    duration_seconds: Optional[float] = Field(None, ge=0)
    error_count: int = Field(0, ge=0)
    video_filename: Optional[str] = None

class ExerciseSessionResponse(BaseModel):
    id: int
    user_id: int
    exercise_id: int
    score: int
    duration_seconds: Optional[float]
    error_count: int
    video_filename: Optional[str]
    created_at: datetime
    
    class Config:
        from_attributes = True

class ExerciseSessionWithDetails(ExerciseSessionResponse):
    user: dict
    exercise: dict


