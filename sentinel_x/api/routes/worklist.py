"""Worklist endpoints."""

import logging
import sys
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse

# Ensure the project root is in path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.models import PRIORITY_COLORS, PRIORITY_NAMES, WorklistEntryResponse, WorklistResponse
from api.services.demo_service import demo_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/worklist", tags=["worklist"])


@router.get("", response_model=WorklistResponse)
async def get_worklist(
    priority: Optional[int] = Query(None, ge=1, le=3, description="Filter by priority level")
):
    """Get all worklist entries, sorted by priority."""
    worklist = demo_service.get_worklist()

    # Reload from disk to ensure we have the latest entries written by agent
    worklist.reload()

    entries = worklist.get_entries(priority_filter=priority)
    stats = worklist.get_statistics()

    logger.info(f"GET /api/worklist -> {len(entries)} entries (priority_filter={priority})")

    # Convert entries to response format
    entry_responses = []
    for entry in entries:
        entry_responses.append(WorklistEntryResponse(
            patient_id=entry.patient_id,
            priority_level=entry.priority_level,
            priority_name=PRIORITY_NAMES.get(entry.priority_level, "UNKNOWN"),
            priority_color=PRIORITY_COLORS.get(entry.priority_level, "#6B7280"),
            findings_summary=entry.findings_summary,
            processed_at=entry.processed_at,
            result_path=entry.result_path,
        ))

    response_data = WorklistResponse(
        entries=entry_responses,
        total=stats["total"],
        by_priority=stats["by_priority"],
        priority_names=stats["priority_names"],
    )

    # Prevent browser caching so page refresh always gets fresh data
    return JSONResponse(
        content=response_data.model_dump(),
        headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"},
    )
