"""
Sentinel-X Triage Agent

An intelligent triage system that monitors incoming CT scans, analyzes them
using MedGemma alongside FHIR clinical context, and produces priority-sorted
worklists for radiologist review.
"""

from .agent import TriageAgent

__all__ = ["TriageAgent"]
