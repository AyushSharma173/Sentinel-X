"""WebSocket connection manager for real-time updates."""

import asyncio
import logging
from typing import Any, Dict, List, Set

from fastapi import WebSocket

from ..models import WSEvent, WSEventType

logger = logging.getLogger(__name__)


class WebSocketManager:
    """Manages WebSocket connections and broadcasts events."""

    def __init__(self):
        self._connections: Set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket) -> None:
        """Accept and register a new WebSocket connection."""
        await websocket.accept()
        async with self._lock:
            self._connections.add(websocket)
        logger.info(f"WebSocket connected. Total connections: {len(self._connections)}")

    async def disconnect(self, websocket: WebSocket) -> None:
        """Remove a WebSocket connection."""
        async with self._lock:
            self._connections.discard(websocket)
        logger.info(f"WebSocket disconnected. Total connections: {len(self._connections)}")

    async def broadcast(self, event: WSEvent) -> None:
        """Broadcast an event to all connected clients."""
        if not self._connections:
            return

        message = event.model_dump_json()

        async with self._lock:
            dead_connections = []

            for connection in self._connections:
                try:
                    await connection.send_text(message)
                except Exception as e:
                    logger.warning(f"Failed to send to WebSocket: {e}")
                    dead_connections.append(connection)

            for conn in dead_connections:
                self._connections.discard(conn)

    async def send_event(
        self,
        event_type: WSEventType,
        data: Dict[str, Any] = None,
    ) -> None:
        """Create and broadcast a WebSocket event."""
        event = WSEvent(event=event_type, data=data or {})
        await self.broadcast(event)

    def get_connection_count(self) -> int:
        """Get the number of active connections."""
        return len(self._connections)


# Global WebSocket manager instance
ws_manager = WebSocketManager()
