from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from app.services.live_detector import router as live_router


from app.config import settings
from app.api.routes import users, exercises, sessions, challenges

app = FastAPI(
    title="Exercise Competition Platform",
    description="Gamified exercise competition platform with form detection",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ====== ADD THIS ======
app.mount(
    "/uploads",
    StaticFiles(directory=Path("uploads")),
    name="uploads"
)
# ======================

# Include routers
app.include_router(users.router, prefix="/api")
app.include_router(exercises.router, prefix="/api")
app.include_router(sessions.router, prefix="/api")
app.include_router(challenges.router, prefix="/api")
app.include_router(live_router)          # NOTE: no prefix= here — the WebSocket
                                          # path /api/live/{exercise} is already
                                          # baked into the router decorators


@app.get("/")
def root():
    return {"message": "Exercise Competition Platform API", "version": "1.0.0"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
