import { useState, useCallback } from 'react';
import type { TriageResult, PatientFHIRContext, VolumeInfo } from '@/types';

const API_BASE = '/api';

export function usePatient() {
  const [triageResult, setTriageResult] = useState<TriageResult | null>(null);
  const [fhirContext, setFhirContext] = useState<PatientFHIRContext | null>(null);
  const [volumeInfo, setVolumeInfo] = useState<VolumeInfo | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchPatient = useCallback(async (patientId: string) => {
    setLoading(true);
    setError(null);

    try {
      // Fetch triage result and FHIR context in parallel
      const [triageResponse, fhirResponse, volumeResponse] = await Promise.all([
        fetch(`${API_BASE}/patients/${patientId}/triage`),
        fetch(`${API_BASE}/patients/${patientId}/fhir`),
        fetch(`${API_BASE}/patients/${patientId}/volume-info`),
      ]);

      // Triage result may not exist yet (queued patient)
      if (triageResponse.ok) {
        const triage: TriageResult = await triageResponse.json();
        setTriageResult(triage);
      } else {
        setTriageResult(null);
      }

      if (fhirResponse.ok) {
        const fhir: PatientFHIRContext = await fhirResponse.json();
        setFhirContext(fhir);
      }

      if (volumeResponse.ok) {
        const volume: VolumeInfo = await volumeResponse.json();
        setVolumeInfo(volume);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch patient data');
    } finally {
      setLoading(false);
    }
  }, []);

  const clearPatient = useCallback(() => {
    setTriageResult(null);
    setFhirContext(null);
    setVolumeInfo(null);
    setError(null);
  }, []);

  const getSliceUrl = useCallback((patientId: string, sliceIndex: number) => {
    return `${API_BASE}/patients/${patientId}/slices/${sliceIndex}`;
  }, []);

  return {
    triageResult,
    fhirContext,
    volumeInfo,
    loading,
    error,
    fetchPatient,
    clearPatient,
    getSliceUrl,
  };
}
