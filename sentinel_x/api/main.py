"""FastAPI application for Sentinel-X demo UI."""

import asyncio
import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Ensure the project root is in path for imports
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.routes import demo, patients, websocket, worklist
from api.services.demo_service import demo_service

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    logger.info("Starting Sentinel-X API server")

    # Set the event loop for the demo service
    demo_service.set_event_loop(asyncio.get_running_loop())

    yield

    # Cleanup on shutdown
    logger.info("Shutting down Sentinel-X API server")

    # Stop demo if running
    if demo_service._status.value == "running":
        await demo_service.stop_demo()


# Create FastAPI app
app = FastAPI(
    title="Sentinel-X API",
    description="AI-powered CT Triage System Demo API",
    version="1.0.0",
    lifespan=lifespan,
)

# Configure CORS for frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Vite dev server
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(demo.router)
app.include_router(worklist.router)
app.include_router(patients.router)
app.include_router(websocket.router)


# Serve static files for production (frontend build)
frontend_dist = Path(__file__).parent.parent / "frontend" / "dist"
if frontend_dist.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dist), html=True), name="frontend")


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "sentinel-x-api"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
