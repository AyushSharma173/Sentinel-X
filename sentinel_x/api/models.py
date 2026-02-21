"""Pydantic models for API responses."""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class PriorityLevel(int, Enum):
    """Priority levels for triage results."""
    CRITICAL = 1
    HIGH_RISK = 2
    ROUTINE = 3


PRIORITY_NAMES = {
    PriorityLevel.CRITICAL: "CRITICAL",
    PriorityLevel.HIGH_RISK: "HIGH RISK",
    PriorityLevel.ROUTINE: "ROUTINE",
}

PRIORITY_COLORS = {
    PriorityLevel.CRITICAL: "#DC2626",  # Red
    PriorityLevel.HIGH_RISK: "#D97706",  # Amber
    PriorityLevel.ROUTINE: "#059669",    # Green
}


class DemoStatus(str, Enum):
    """Demo system status."""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    COMPLETED = "completed"


class SystemStatus(BaseModel):
    """Overall system status response."""
    demo_status: DemoStatus = DemoStatus.STOPPED
    simulator_running: bool = False
    agent_running: bool = False
    model_loaded: bool = False
    patients_in_queue: int = 0
    patients_processed: int = 0


class WorklistEntryResponse(BaseModel):
    """Single worklist entry response."""
    patient_id: str
    priority_level: int
    priority_name: str
    priority_color: str
    findings_summary: str
    processed_at: str
    result_path: str


class WorklistResponse(BaseModel):
    """Worklist response with entries and statistics."""
    entries: List[WorklistEntryResponse]
    total: int
    by_priority: Dict[int, int]
    priority_names: Dict[int, str]


class PatientDemographics(BaseModel):
    """Patient demographic information from FHIR."""
    patient_id: str
    age: Optional[int] = None
    gender: Optional[str] = None


class PatientCondition(BaseModel):
    """Patient medical condition."""
    name: str
    is_risk_factor: bool = False


class PatientFHIRContext(BaseModel):
    """Full FHIR context for a patient."""
    patient_id: str
    demographics: PatientDemographics
    conditions: List[PatientCondition]
    medications: List[str]
    risk_factors: List[str]
    findings: str
    impressions: str


class DeltaAnalysisEntry(BaseModel):
    """A single finding classification from Delta Analysis."""
    finding: str
    classification: str
    priority: int
    history_match: Optional[str] = None
    reasoning: str


class TriageResult(BaseModel):
    """Full triage result for a patient."""
    patient_id: str
    priority_level: int
    priority_name: str
    priority_color: str
    rationale: str
    key_slice_index: int
    key_slice_thumbnail: str
    processed_at: str
    conditions_considered: List[str]
    findings_summary: str
    visual_findings: str
    # Serial Late Fusion fields
    delta_analysis: List[DeltaAnalysisEntry] = Field(default_factory=list)
    phase1_raw: str = ""
    phase2_raw: str = ""
    headline: str = ""
    reasoning: str = ""


class QueuedPatientResponse(BaseModel):
    """A single patient in the queue."""
    patient_id: str
    status: str  # "queued" | "processing"
    phase: Optional[str] = None  # "phase1" | "model_swap" | "phase2"


class QueueStateResponse(BaseModel):
    """Current queue state for UI recovery after page refresh."""
    patients: List[QueuedPatientResponse]


class DemoControlResponse(BaseModel):
    """Response for demo control operations."""
    success: bool
    message: str
    status: SystemStatus


class WSEventType(str, Enum):
    """WebSocket event types."""
    DEMO_STARTED = "demo_started"
    DEMO_STOPPED = "demo_stopped"
    PATIENT_ARRIVED = "patient_arrived"
    PROCESSING_STARTED = "processing_started"
    PROCESSING_PROGRESS = "processing_progress"
    PROCESSING_COMPLETE = "processing_complete"
    WORKLIST_UPDATED = "worklist_updated"
    ERROR = "error"
    # Serial Late Fusion phase events
    PHASE1_STARTED = "phase1_started"
    PHASE1_COMPLETE = "phase1_complete"
    MODEL_SWAPPING = "model_swapping"
    PHASE2_STARTED = "phase2_started"
    PHASE2_COMPLETE = "phase2_complete"
    DEMO_COMPLETE = "demo_complete"


class WSEvent(BaseModel):
    """WebSocket event message."""
    event: WSEventType
    data: Dict[str, Any] = Field(default_factory=dict)
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
