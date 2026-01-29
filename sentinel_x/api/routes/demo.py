"""Demo control endpoints."""

import sys
from pathlib import Path

from fastapi import APIRouter, HTTPException

# Ensure the project root is in path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.models import DemoControlResponse, SystemStatus
from api.services.demo_service import demo_service

router = APIRouter(prefix="/api/demo", tags=["demo"])


@router.get("/status", response_model=SystemStatus)
async def get_status():
    """Get current demo status."""
    return demo_service.get_status()


@router.post("/start", response_model=DemoControlResponse)
async def start_demo():
    """Start the demo (simulator + triage agent)."""
    success = await demo_service.start_demo()

    if not success:
        raise HTTPException(
            status_code=409,
            detail="Demo is already running or failed to start"
        )

    return DemoControlResponse(
        success=True,
        message="Demo started successfully",
        status=demo_service.get_status()
    )


@router.post("/stop", response_model=DemoControlResponse)
async def stop_demo():
    """Stop the demo."""
    success = await demo_service.stop_demo()

    if not success:
        raise HTTPException(
            status_code=409,
            detail="Demo is not running"
        )

    return DemoControlResponse(
        success=True,
        message="Demo stopped successfully",
        status=demo_service.get_status()
    )


@router.post("/reset", response_model=DemoControlResponse)
async def reset_demo():
    """Reset the demo (clear inbox and worklist)."""
    success = await demo_service.reset_demo()

    return DemoControlResponse(
        success=True,
        message="Demo reset successfully",
        status=demo_service.get_status()
    )
