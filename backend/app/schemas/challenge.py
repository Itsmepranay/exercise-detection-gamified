from pydantic import BaseModel
from datetime import datetime
from typing import Optional

class ChallengeCreate(BaseModel):
    challenger_id: int
    opponent_id: int
    exercise_id: int

class ChallengeSubmit(BaseModel):
    exercise_session_id: int

class ChallengeResponse(BaseModel):
    id: int
    challenger_id: int
    opponent_id: int
    exercise_id: int
    status: str
    winner_id: Optional[int]
    created_at: datetime
    completed_at: Optional[datetime]
    
    class Config:
        from_attributes = True

class ChallengeWithDetails(ChallengeResponse):
    challenger: dict
    opponent: dict
    exercise: dict
    winner: Optional[dict] = None
    sessions: list = []


