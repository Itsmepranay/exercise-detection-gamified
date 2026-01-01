#!/usr/bin/env python3
"""
Script to update winner_id for completed challenges that don't have a winner set.
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from app.database import get_db
from app.services.challenge_service import determine_challenge_winner
from app.models.challenge import Challenge
from sqlalchemy.orm import Session

def update_winners():
    db: Session = next(get_db())
    try:
        # Get all completed challenges without a winner
        challenges = db.query(Challenge).filter(
            Challenge.status == 'completed',
            Challenge.winner_id.is_(None)
        ).all()

        print(f"Found {len(challenges)} completed challenges without winner")

        for challenge in challenges:
            winner_id = determine_challenge_winner(challenge, db)
            if winner_id:
                challenge.winner_id = winner_id
                print(f"Updated challenge {challenge.id}: winner {winner_id}")
            else:
                print(f"Challenge {challenge.id}: still no winner (tie or insufficient data)")

        db.commit()
        print("All updates committed")

    except Exception as e:
        print(f"Error: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    update_winners()