from sqlalchemy.orm import Session
from app.models.challenge import Challenge, ChallengeSession, ChallengeStatus
from app.models.session import ExerciseSession
from datetime import datetime

def determine_challenge_winner(challenge: Challenge, db: Session) -> int | None:
    """
    Determine winner of challenge based on best scores.
    Returns user_id of winner, or None if tied/not enough submissions.
    """
    # Get all sessions submitted for this challenge
    challenge_sessions = db.query(ChallengeSession).filter(
        ChallengeSession.challenge_id == challenge.id
    ).all()
    
    if len(challenge_sessions) < 2:
        # Need at least one submission from each participant
        return None
    
    session_ids = [cs.exercise_session_id for cs in challenge_sessions]
    sessions = db.query(ExerciseSession).filter(
        ExerciseSession.id.in_(session_ids)
    ).all()
    
    # Get best score for each participant
    challenger_best = None
    opponent_best = None
    
    for session in sessions:
        if session.user_id == challenge.challenger_id:
            if challenger_best is None or session.score > challenger_best.score:
                challenger_best = session
        elif session.user_id == challenge.opponent_id:
            if opponent_best is None or session.score > opponent_best.score:
                opponent_best = session
    
    if not challenger_best or not opponent_best:
        return None
    
    # Determine winner (higher score wins)
    if challenger_best.score > opponent_best.score:
        return challenge.challenger_id
    elif opponent_best.score > challenger_best.score:
        return challenge.opponent_id
    else:
        # Tie - could return None or use tiebreaker (e.g., fewer errors)
        return None

def update_challenge_status(challenge: Challenge, db: Session):
    """Update challenge status based on submissions"""
    challenge_sessions = db.query(ChallengeSession).filter(
        ChallengeSession.challenge_id == challenge.id
    ).all()
    
    session_ids = [cs.exercise_session_id for cs in challenge_sessions]
    sessions = db.query(ExerciseSession).filter(
        ExerciseSession.id.in_(session_ids)
    ).all() if session_ids else []
    
    challenger_submitted = any(s.user_id == challenge.challenger_id for s in sessions)
    opponent_submitted = any(s.user_id == challenge.opponent_id for s in sessions)
    
    if challenger_submitted and opponent_submitted:
        # Both submitted - determine winner and mark as completed
        challenge.status = ChallengeStatus.COMPLETED
        challenge.completed_at = datetime.utcnow()
        winner_id = determine_challenge_winner(challenge, db)
        if winner_id:
            challenge.winner_id = winner_id
    elif challenger_submitted or opponent_submitted:
        # One submitted - mark as active
        challenge.status = ChallengeStatus.ACTIVE
    # else: remains PENDING

