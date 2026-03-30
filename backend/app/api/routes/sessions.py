from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.orm import Session
from typing import Optional
import uuid
from pathlib import Path
from app.database import get_db
from app.models.session import ExerciseSession
from app.models.exercise import Exercise
from app.models.user import User
from app.schemas.session import ExerciseSessionCreate, ExerciseSessionResponse, ExerciseSessionWithDetails
from app.services.exercise_detector import get_detector
from app.utils.video_processor import save_uploaded_video

router = APIRouter(prefix="/sessions", tags=["sessions"])

# Base URL the frontend uses to reach the FastAPI server.
# Videos are served via app.mount("/uploads", StaticFiles(...)) in main.py.
# The frontend constructs:  API_BASE_URL + /uploads/annotated/<filename>
# which maps to:            uploads/annotated/<filename> on disk.
_API_SERVER = "http://127.0.0.1:8080"


@router.post("/upload", status_code=201)
async def upload_video(
    user_id: int = Form(...),
    exercise_id: int = Form(...),
    video: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """Upload video, run exercise detection, save annotated video, create session."""

    # ── Validate user & exercise ───────────────────────────────────────────
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    exercise = db.query(Exercise).filter(Exercise.id == exercise_id).first()
    if not exercise:
        raise HTTPException(status_code=404, detail="Exercise not found")

    # ── Generate filenames ─────────────────────────────────────────────────
    base_id            = uuid.uuid4()
    file_extension     = Path(video.filename).suffix if video.filename else ".mp4"
    video_filename     = f"{base_id}{file_extension}"
    annotated_filename = f"{base_id}_annotated.mp4"

    try:
        # ── Save original upload ───────────────────────────────────────────
        video_path = save_uploaded_video(video, video_filename)

        # ── Annotated video output path ────────────────────────────────────
        # Saved to:  uploads/annotated/<uuid>_annotated.mp4
        # Served at: /uploads/annotated/<uuid>_annotated.mp4  (StaticFiles mount)
        annotated_dir  = Path("uploads") / "annotated"
        annotated_dir.mkdir(parents=True, exist_ok=True)
        annotated_path = str(annotated_dir / annotated_filename)

        # ── Run detection + annotation ─────────────────────────────────────
        detector = get_detector(exercise.name)
        results  = detector.process_video(video_path, output_path=annotated_path)

        # ── Build the public URL for the annotated video ───────────────────
        # results["processed_video"] is just the filename (or None on failure).
        # We build the full URL here so the frontend only needs to display it.
        processed_video_url = None
        if results.get("processed_video"):
            processed_video_url = (
                f"{_API_SERVER}/uploads/annotated/{results['processed_video']}"
            )

        # ── Persist session ────────────────────────────────────────────────
        session = ExerciseSession(
            user_id          = user_id,
            exercise_id      = exercise_id,
            score            = results["score"],
            duration_seconds = results["duration_seconds"],
            error_count      = results["error_count"],
            video_filename   = video_filename,
        )
        db.add(session)
        db.commit()
        db.refresh(session)

        # ── Return response ────────────────────────────────────────────────
        return {
            "id"               : session.id,
            "user_id"          : session.user_id,
            "exercise_id"      : session.exercise_id,
            "score"            : session.score,
            "duration_seconds" : session.duration_seconds,
            "error_count"      : session.error_count,
            "video_filename"   : session.video_filename,
            "processed_video"  : processed_video_url,   # full URL or None
            "created_at"       : session.created_at,
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing video: {str(e)}"
        )


@router.post("", response_model=ExerciseSessionResponse, status_code=201)
def create_session(session_data: ExerciseSessionCreate, db: Session = Depends(get_db)):
    """Create exercise session directly (for testing or manual entry)."""
    user = db.query(User).filter(User.id == session_data.user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    exercise = db.query(Exercise).filter(Exercise.id == session_data.exercise_id).first()
    if not exercise:
        raise HTTPException(status_code=404, detail="Exercise not found")

    session = ExerciseSession(**session_data.model_dump())
    db.add(session)
    db.commit()
    db.refresh(session)
    return session


@router.get("/{session_id}", response_model=ExerciseSessionResponse)
def get_session(session_id: int, db: Session = Depends(get_db)):
    """Get session by ID."""
    session = db.query(ExerciseSession).filter(ExerciseSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


@router.get("/user/{user_id}", response_model=list[ExerciseSessionWithDetails])
def get_user_sessions(
    user_id: int,
    exercise_id: Optional[int] = None,
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """Get all sessions for a user."""
    query = db.query(ExerciseSession).filter(ExerciseSession.user_id == user_id)
    if exercise_id:
        query = query.filter(ExerciseSession.exercise_id == exercise_id)

    sessions = query.order_by(
        ExerciseSession.created_at.desc()
    ).offset(skip).limit(limit).all()

    result = []
    for session in sessions:
        result.append({
            **session.__dict__,
            "user"    : {"id": session.user.id,     "username": session.user.username},
            "exercise": {
                "id"          : session.exercise.id,
                "name"        : session.exercise.name,
                "display_name": session.exercise.display_name,
            },
        })
    return result