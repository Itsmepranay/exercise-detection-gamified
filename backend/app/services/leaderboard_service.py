from sqlalchemy.orm import Session
from sqlalchemy import func, desc
from app.models.session import ExerciseSession
from app.models.user import User
from datetime import datetime, timedelta
from typing import Optional, List, Dict

def get_leaderboard(
    exercise_id: int,
    period: str = "all_time",
    limit: int = 10,
    db: Session = None
) -> List[Dict]:
    """
    Get leaderboard for an exercise.
    
    Args:
        exercise_id: ID of exercise
        period: "all_time", "weekly", "daily"
        limit: Number of top entries to return
        db: Database session
    
    Returns:
        List of dicts with user info and best score
    """
    query = db.query(
        ExerciseSession.user_id,
        User.username,
        func.max(ExerciseSession.score).label('best_score')
    ).join(
        User, ExerciseSession.user_id == User.id
    ).filter(
        ExerciseSession.exercise_id == exercise_id
    )
    
    # Apply time filter
    if period == "weekly":
        week_ago = datetime.utcnow() - timedelta(days=7)
        query = query.filter(ExerciseSession.created_at >= week_ago)
    elif period == "daily":
        today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        query = query.filter(ExerciseSession.created_at >= today)
    # all_time: no filter
    
    # Group by user and get best score
    results = query.group_by(
        ExerciseSession.user_id, User.username
    ).order_by(
        desc('best_score')
    ).limit(limit).all()
    
    leaderboard = []
    rank = 1
    for user_id, username, best_score in results:
        leaderboard.append({
            "rank": rank,
            "user_id": user_id,
            "username": username,
            "best_score": int(best_score)
        })
        rank += 1
    
    return leaderboard

def get_user_rank(
    user_id: int,
    exercise_id: int,
    period: str = "all_time",
    db: Session = None
) -> Optional[Dict]:
    """Get user's rank for an exercise"""
    leaderboard = get_leaderboard(exercise_id, period, limit=1000, db=db)
    
    for entry in leaderboard:
        if entry["user_id"] == user_id:
            return entry
    
    return None


