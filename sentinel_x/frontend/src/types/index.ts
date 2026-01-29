/**
 * TypeScript type definitions for Sentinel-X frontend
 */

export type DemoStatus = 'stopped' | 'starting' | 'running' | 'stopping';

export interface SystemStatus {
  demo_status: DemoStatus;
  simulator_running: boolean;
  agent_running: boolean;
  model_loaded: boolean;
  patients_in_queue: number;
  patients_processed: number;
}

export interface WorklistEntry {
  patient_id: string;
  priority_level: number;
  priority_name: string;
  priority_color: string;
  findings_summary: string;
  processed_at: string;
  result_path: string;
}

export interface WorklistResponse {
  entries: WorklistEntry[];
  total: number;
  by_priority: Record<number, number>;
  priority_names: Record<number, string>;
}

export interface PatientDemographics {
  patient_id: string;
  age: number | null;
  gender: string | null;
}

export interface PatientCondition {
  name: string;
  is_risk_factor: boolean;
}

export interface PatientFHIRContext {
  patient_id: string;
  demographics: PatientDemographics;
  conditions: PatientCondition[];
  medications: string[];
  risk_factors: string[];
  findings: string;
  impressions: string;
}

export interface TriageResult {
  patient_id: string;
  priority_level: number;
  priority_name: string;
  priority_color: string;
  rationale: string;
  key_slice_index: number;
  key_slice_thumbnail: string;
  processed_at: string;
  conditions_considered: string[];
  findings_summary: string;
  visual_findings: string;
}

export interface DemoControlResponse {
  success: boolean;
  message: string;
  status: SystemStatus;
}

export type WSEventType =
  | 'demo_started'
  | 'demo_stopped'
  | 'patient_arrived'
  | 'processing_started'
  | 'processing_progress'
  | 'processing_complete'
  | 'worklist_updated'
  | 'error';

export interface WSEvent {
  event: WSEventType;
  data: Record<string, unknown>;
  timestamp: string;
}

export interface VolumeInfo {
  patient_id: string;
  total_slices: number;
  dimensions: number[];
}
