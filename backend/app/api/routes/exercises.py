from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import Optional
from app.database import get_db
from app.models.exercise import Exercise
from app.schemas.exercise import ExerciseResponse
from app.services.leaderboard_service import get_leaderboard

router = APIRouter(prefix="/exercises", tags=["exercises"])

@router.get("", response_model=list[ExerciseResponse])
def list_exercises(db: Session = Depends(get_db)):
    """List all available exercises"""
    exercises = db.query(Exercise).all()
    return exercises

@router.get("/{exercise_id}", response_model=ExerciseResponse)
def get_exercise(exercise_id: int, db: Session = Depends(get_db)):
    """Get exercise by ID"""
    exercise = db.query(Exercise).filter(Exercise.id == exercise_id).first()
    if not exercise:
        raise HTTPException(status_code=404, detail="Exercise not found")
    return exercise

@router.get("/{exercise_id}/leaderboard")
def get_exercise_leaderboard(
    exercise_id: int,
    period: str = "all_time",
    limit: int = 10,
    db: Session = Depends(get_db)
):
    """Get leaderboard for an exercise"""
    # Validate exercise exists
    exercise = db.query(Exercise).filter(Exercise.id == exercise_id).first()
    if not exercise:
        raise HTTPException(status_code=404, detail="Exercise not found")
    
    if period not in ["all_time", "weekly", "daily"]:
        raise HTTPException(status_code=400, detail="Invalid period. Use: all_time, weekly, daily")
    
    leaderboard = get_leaderboard(exercise_id, period, limit, db)
    return {"exercise_id": exercise_id, "period": period, "leaderboard": leaderboard}

