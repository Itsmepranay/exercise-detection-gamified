#!/usr/bin/env python3
"""
Script to initialize the database with default exercises
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from app.database import SessionLocal
from app.models.exercise import Exercise

def init_exercises():
    db = SessionLocal()
    try:
        # Check if exercises already exist
        existing = db.query(Exercise).count()
        if existing > 0:
            print(f"Exercises already exist ({existing} found). Skipping initialization.")
            return

        # Create default exercises
        exercises = [
            Exercise(
                name="bicep_curl",
                display_name="Bicep Curl",
                description="Perform bicep curls with proper form detection",
                metric_type="reps"
            ),
            Exercise(
                name="plank",
                display_name="Plank",
                description="Hold a plank position with form analysis",
                metric_type="duration"
            ),
            Exercise(
                name="squat",
                display_name="Squat",
                description="Perform squats with knee angle detection",
                metric_type="reps"
            )
        ]

        for exercise in exercises:
            db.add(exercise)

        db.commit()
        print(f"Successfully initialized {len(exercises)} exercises")

    except Exception as e:
        print(f"Error initializing exercises: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    init_exercises()