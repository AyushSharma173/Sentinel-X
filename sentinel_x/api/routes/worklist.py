"""Worklist endpoints."""

import sys
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Query

# Ensure the project root is in path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.models import PRIORITY_COLORS, PRIORITY_NAMES, WorklistEntryResponse, WorklistResponse
from api.services.demo_service import demo_service

router = APIRouter(prefix="/api/worklist", tags=["worklist"])


@router.get("", response_model=WorklistResponse)
async def get_worklist(
    priority: Optional[int] = Query(None, ge=1, le=3, description="Filter by priority level")
):
    """Get all worklist entries, sorted by priority."""
    worklist = demo_service.get_worklist()

    entries = worklist.get_entries(priority_filter=priority)
    stats = worklist.get_statistics()

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

    return WorklistResponse(
        entries=entry_responses,
        total=stats["total"],
        by_priority=stats["by_priority"],
        priority_names=stats["priority_names"],
    )
