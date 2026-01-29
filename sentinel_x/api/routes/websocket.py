"""WebSocket endpoint for real-time updates."""

import logging
import sys
from pathlib import Path

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

# Ensure the project root is in path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.services.ws_manager import ws_manager

logger = logging.getLogger(__name__)

router = APIRouter(tags=["websocket"])


@router.websocket("/ws/triage")
async def triage_websocket(websocket: WebSocket):
    """WebSocket endpoint for real-time triage updates."""
    await ws_manager.connect(websocket)

    try:
        while True:
            # Keep connection alive by waiting for messages
            # (mainly used for ping/pong and client disconnect detection)
            data = await websocket.receive_text()

            # Handle ping messages
            if data == "ping":
                await websocket.send_text("pong")

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        await ws_manager.disconnect(websocket)
