import os
import shutil
from pathlib import Path
from typing import Optional
from app.config import settings

def save_uploaded_video(file, filename: str) -> str:
    """
    Save uploaded video file to storage directory.
    Returns the saved file path.
    """
    # Ensure video directory exists
    video_dir = Path(settings.video_dir)
    video_dir.mkdir(parents=True, exist_ok=True)
    
    # Create full path
    file_path = video_dir / filename
    
    # Save file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    return str(file_path)

def get_video_path(filename: str) -> Optional[str]:
    """Get full path to video file if it exists"""
    file_path = Path(settings.video_dir) / filename
    if file_path.exists():
        return str(file_path)
    return None

def delete_video(filename: str) -> bool:
    """Delete video file. Returns True if deleted, False if not found"""
    file_path = Path(settings.video_dir) / filename
    if file_path.exists():
        file_path.unlink()
        return True
    return False

