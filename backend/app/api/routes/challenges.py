from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session, selectinload
from typing import Optional
from datetime import datetime
from app.database import get_db
from app.models.challenge import Challenge, ChallengeSession
from app.models.session import ExerciseSession
from app.models.exercise import Exercise
from app.models.user import User
from app.schemas.challenge import ChallengeCreate, ChallengeResponse, ChallengeSubmit, ChallengeWithDetails, ChallengeWithDetails
from app.services.challenge_service import determine_challenge_winner, update_challenge_status

router = APIRouter(prefix="/challenges", tags=["challenges"])

@router.post("", response_model=ChallengeResponse, status_code=201)
def create_challenge(challenge_data: ChallengeCreate, db: Session = Depends(get_db)):
    """Create a new challenge"""
    # Validate challenger exists
    challenger = db.query(User).filter(User.id == challenge_data.challenger_id).first()
    if not challenger:
        raise HTTPException(status_code=404, detail="Challenger not found")
    
    # Validate opponent exists
    opponent = db.query(User).filter(User.id == challenge_data.opponent_id).first()
    if not opponent:
        raise HTTPException(status_code=404, detail="Opponent not found")
    
    # Cannot challenge yourself
    if challenge_data.challenger_id == challenge_data.opponent_id:
        raise HTTPException(status_code=400, detail="Cannot challenge yourself")
    
    # Validate exercise exists
    exercise = db.query(Exercise).filter(Exercise.id == challenge_data.exercise_id).first()
    if not exercise:
        raise HTTPException(status_code=404, detail="Exercise not found")
    
    challenge = Challenge(
        challenger_id=challenge_data.challenger_id,
        opponent_id=challenge_data.opponent_id,
        exercise_id=challenge_data.exercise_id,
        status="pending"
    )
    
    db.add(challenge)
    db.commit()
    db.refresh(challenge)
    return challenge

@router.get("/{challenge_id}", response_model=ChallengeWithDetails)
def get_challenge(challenge_id: int, db: Session = Depends(get_db)):
    """Get challenge details"""
    challenge = db.query(Challenge).filter(Challenge.id == challenge_id).first()
    if not challenge:
        raise HTTPException(status_code=404, detail="Challenge not found")
    
    # Get related sessions
    challenge_sessions = db.query(ChallengeSession).filter(
        ChallengeSession.challenge_id == challenge_id
    ).all()
    
    session_ids = [cs.exercise_session_id for cs in challenge_sessions]
    sessions = db.query(ExerciseSession).filter(
        ExerciseSession.id.in_(session_ids)
    ).all() if session_ids else []
    
    return {
        **challenge.__dict__,
        "challenger": {"id": challenge.challenger.id, "username": challenge.challenger.username},
        "opponent": {"id": challenge.opponent.id, "username": challenge.opponent.username},
        "exercise": {"id": challenge.exercise.id, "name": challenge.exercise.name, "display_name": challenge.exercise.display_name},
        "winner": {"id": challenge.winner.id, "username": challenge.winner.username} if challenge.winner else None,
        "sessions": [{"id": s.id, "user_id": s.user_id, "score": s.score} for s in sessions]
    }

@router.post("/{challenge_id}/submit", response_model=ChallengeResponse)
def submit_to_challenge(
    challenge_id: int,
    submit_data: ChallengeSubmit,
    db: Session = Depends(get_db)
):
    """Submit a session for a challenge"""
    challenge = db.query(Challenge).filter(Challenge.id == challenge_id).first()
    if not challenge:
        raise HTTPException(status_code=404, detail="Challenge not found")
    
    # Check challenge status
    if challenge.status == "completed":
        raise HTTPException(status_code=400, detail="Challenge already completed")
    if challenge.status == "rejected":
        raise HTTPException(status_code=400, detail="Challenge was rejected")
    
    # Validate session exists and belongs to challenge exercise
    session = db.query(ExerciseSession).filter(
        ExerciseSession.id == submit_data.exercise_session_id
    ).first()
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    if session.exercise_id != challenge.exercise_id:
        raise HTTPException(status_code=400, detail="Session exercise does not match challenge exercise")
    
    # Verify user is part of challenge
    if session.user_id not in [challenge.challenger_id, challenge.opponent_id]:
        raise HTTPException(status_code=403, detail="User not part of this challenge")
    
    # Check if session already submitted
    existing = db.query(ChallengeSession).filter(
        ChallengeSession.challenge_id == challenge_id,
        ChallengeSession.exercise_session_id == submit_data.exercise_session_id
    ).first()
    
    if existing:
        raise HTTPException(status_code=400, detail="Session already submitted to this challenge")
    
    # Link session to challenge
    challenge_session = ChallengeSession(
        challenge_id=challenge_id,
        exercise_session_id=submit_data.exercise_session_id
    )
    db.add(challenge_session)
    db.commit()
    
    # Update challenge status and determine winner
    update_challenge_status(challenge, db)
    
    db.commit()
    db.refresh(challenge)
    return challenge

@router.post("/{challenge_id}/accept", response_model=ChallengeResponse)
def accept_challenge(
    challenge_id: int,
    db: Session = Depends(get_db)
):
    """Accept a challenge (for the opponent)"""
    # Note: In a real app, you'd verify the current user is the opponent
    challenge = db.query(Challenge).filter(Challenge.id == challenge_id).first()
    if not challenge:
        raise HTTPException(status_code=404, detail="Challenge not found")
    
    if challenge.status != "pending":
        raise HTTPException(status_code=400, detail="Challenge is not pending")
    
    # Change status to active to indicate it's accepted and ready
    challenge.status = "active"
    db.commit()
    return challenge

@router.post("/{challenge_id}/reject", response_model=ChallengeResponse)
def reject_challenge(
    challenge_id: int,
    db: Session = Depends(get_db)
):
    """Reject a challenge (for the opponent)"""
    # Note: In a real app, you'd verify the current user is the opponent
    challenge = db.query(Challenge).filter(Challenge.id == challenge_id).first()
    if not challenge:
        raise HTTPException(status_code=404, detail="Challenge not found")
    
    if challenge.status != "pending":
        raise HTTPException(status_code=400, detail="Challenge cannot be rejected")
    
    challenge.status = "rejected"
    db.commit()
    return challenge

@router.get("/user/{user_id}", response_model=list[ChallengeWithDetails])
def get_user_challenges(
    user_id: int,
    status: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Get challenges for a user"""
    query = db.query(Challenge).filter(
        (Challenge.challenger_id == user_id) | (Challenge.opponent_id == user_id)
    )
    
    if status:
        query = query.filter(Challenge.status == status)
    
    challenges = query.options(
        selectinload(Challenge.challenger),
        selectinload(Challenge.opponent),
        selectinload(Challenge.winner),
        selectinload(Challenge.exercise),
        selectinload(Challenge.challenge_sessions).selectinload(ChallengeSession.exercise_session)
    ).order_by(Challenge.created_at.desc()).all()
    
    # Build detailed response
    result = []
    for challenge in challenges:
        challenge_dict = {
            "id": challenge.id,
            "challenger_id": challenge.challenger_id,
            "opponent_id": challenge.opponent_id,
            "exercise_id": challenge.exercise_id,
            "status": challenge.status,
            "winner_id": challenge.winner_id,
            "created_at": challenge.created_at,
            "completed_at": challenge.completed_at,
            "challenger": {
                "id": challenge.challenger.id,
                "username": challenge.challenger.username
            },
            "opponent": {
                "id": challenge.opponent.id,
                "username": challenge.opponent.username
            },
            "exercise": {
                "id": challenge.exercise.id,
                "name": challenge.exercise.name,
                "display_name": challenge.exercise.display_name
            },
            "winner": {
                "id": challenge.winner.id,
                "username": challenge.winner.username
            } if challenge.winner else None,
            "sessions": [
                {
                    "id": cs.exercise_session.id,
                    "user_id": cs.exercise_session.user_id,
                    "score": cs.exercise_session.score,
                    "duration_seconds": cs.exercise_session.duration_seconds,
                    "error_count": cs.exercise_session.error_count,
                    "created_at": cs.exercise_session.created_at
                } for cs in challenge.challenge_sessions
            ]
        }
        result.append(challenge_dict)
    
    return result

