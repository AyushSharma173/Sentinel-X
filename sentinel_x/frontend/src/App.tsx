import { useState, useCallback } from 'react';
import { Wifi, WifiOff } from 'lucide-react';
import { Dashboard } from '@/components/dashboard';
import { WorklistTable, ProcessingIndicator } from '@/components/worklist';
import { PatientDetail } from '@/components/patient';
import { useDemoControls } from '@/hooks/useDemoControls';
import { useWorklist } from '@/hooks/useWorklist';
import { useTriageWebSocket } from '@/hooks/useTriageWebSocket';
import { usePatient } from '@/hooks/usePatient';
import type { SystemStatus, QueuedPatient } from '@/types';

function App() {
  const [selectedPatientId, setSelectedPatientId] = useState<string | null>(null);
  const [processingPatientId, setProcessingPatientId] = useState<string | null>(null);
  const [newPatientIds, setNewPatientIds] = useState<Set<string>>(new Set());
  const [queuedPatients, setQueuedPatients] = useState<QueuedPatient[]>([]);
  const [currentPhase, setCurrentPhase] = useState<string | null>(null);

  // Demo controls
  const {
    status,
    loading: demoLoading,
    startDemo,
    stopDemo,
    resetDemo,
    updateStatus,
  } = useDemoControls();

  // Worklist
  const {
    entries,
    stats,
    loading: worklistLoading,
    priorityFilter,
    setFilter,
    refresh: refreshWorklist,
  } = useWorklist();

  // Patient detail
  const {
    triageResult,
    fhirContext,
    volumeInfo,
    loading: patientLoading,
    error: patientError,
    fetchPatient,
    clearPatient,
    getSliceUrl,
  } = usePatient();

  // WebSocket for real-time updates
  const { connected } = useTriageWebSocket({
    onDemoStarted: (data) => {
      const newStatus = data.status as SystemStatus;
      if (newStatus) updateStatus(newStatus);
    },
    onDemoStopped: (data) => {
      const newStatus = data.status as SystemStatus;
      if (newStatus) updateStatus(newStatus);
      setProcessingPatientId(null);
      setCurrentPhase(null);
    },
    onPatientArrived: (data) => {
      const patientId = data.patient_id as string;
      setQueuedPatients((prev) => {
        // Deduplicate
        if (prev.some((qp) => qp.patient_id === patientId)) return prev;
        return [...prev, {
          patient_id: patientId,
          status: 'queued',
          arrived_at: new Date().toISOString(),
        }];
      });
    },
    onProcessingStarted: (data) => {
      const patientId = data.patient_id as string;
      setProcessingPatientId(patientId);
      setCurrentPhase('phase1');
      // Update queued patient to analyzing
      setQueuedPatients((prev) =>
        prev.map((qp) =>
          qp.patient_id === patientId
            ? { ...qp, status: 'analyzing' as const, phase: 'phase1' as const }
            : qp
        )
      );
    },
    onPhase1Complete: () => {
      setCurrentPhase('model_swap');
      setQueuedPatients((prev) =>
        prev.map((qp) =>
          qp.status === 'analyzing'
            ? { ...qp, phase: 'model_swap' as const }
            : qp
        )
      );
    },
    onPhase2Started: () => {
      setCurrentPhase('phase2');
      setQueuedPatients((prev) =>
        prev.map((qp) =>
          qp.status === 'analyzing'
            ? { ...qp, phase: 'phase2' as const }
            : qp
        )
      );
    },
    onProcessingComplete: (data) => {
      const patientId = data.patient_id as string;
      setProcessingPatientId(null);
      setCurrentPhase(null);

      // Remove from queued patients
      setQueuedPatients((prev) => prev.filter((qp) => qp.patient_id !== patientId));

      // Add to new patients set for highlight animation
      setNewPatientIds((prev) => new Set(prev).add(patientId));

      // Remove from new patients after animation
      setTimeout(() => {
        setNewPatientIds((prev) => {
          const next = new Set(prev);
          next.delete(patientId);
          return next;
        });
      }, 3000);

      // Refresh worklist
      refreshWorklist();
    },
    onWorklistUpdated: () => {
      refreshWorklist();
    },
    onDemoComplete: (data) => {
      const newStatus = data.status as SystemStatus;
      if (newStatus) updateStatus(newStatus);
      setProcessingPatientId(null);
      setCurrentPhase(null);
    },
  });

  // Handle patient selection
  const handlePatientClick = useCallback((patientId: string) => {
    setSelectedPatientId(patientId);
    fetchPatient(patientId);
  }, [fetchPatient]);

  // Handle patient detail close
  const handleCloseDetail = useCallback(() => {
    setSelectedPatientId(null);
    clearPatient();
  }, [clearPatient]);

  // Handle reset
  const handleReset = useCallback(async () => {
    await resetDemo();
    refreshWorklist();
    setQueuedPatients([]);
    setCurrentPhase(null);
  }, [resetDemo, refreshWorklist]);

  // Determine if we should show the worklist view
  const showWorklist = status?.demo_status === 'running' || status?.demo_status === 'completed' || entries.length > 0 || queuedPatients.length > 0;

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Connection status indicator */}
      <div className="fixed top-4 right-4 z-40">
        <div
          className={`flex items-center gap-2 px-3 py-1.5 rounded-full text-sm ${
            connected
              ? 'bg-routine/10 text-routine'
              : 'bg-gray-100 text-gray-500'
          }`}
        >
          {connected ? (
            <>
              <Wifi className="h-4 w-4" />
              <span>Live</span>
            </>
          ) : (
            <>
              <WifiOff className="h-4 w-4" />
              <span>Disconnected</span>
            </>
          )}
        </div>
      </div>

      {/* Main content */}
      <main className="pb-8">
        {/* Dashboard */}
        <Dashboard
          status={status}
          loading={demoLoading}
          stats={stats}
          onStart={startDemo}
          onStop={stopDemo}
          onReset={handleReset}
        />

        {/* Worklist section */}
        {showWorklist && (
          <div className="container mx-auto px-4 max-w-6xl mt-8">
            <WorklistTable
              entries={entries}
              loading={worklistLoading}
              priorityFilter={priorityFilter}
              onFilterChange={setFilter}
              onRowClick={handlePatientClick}
              counts={stats.byPriority}
              newPatientIds={newPatientIds}
              queuedPatients={queuedPatients}
            />
          </div>
        )}
      </main>

      {/* Patient detail slide-out */}
      {selectedPatientId && (
        <>
          {/* Backdrop */}
          <div
            className="fixed inset-0 bg-black/20 z-40"
            onClick={handleCloseDetail}
          />
          <PatientDetail
            triageResult={triageResult}
            fhirContext={fhirContext}
            volumeInfo={volumeInfo}
            loading={patientLoading}
            error={patientError}
            onClose={handleCloseDetail}
            getSliceUrl={getSliceUrl}
            selectedPatientId={selectedPatientId}
          />
        </>
      )}

      {/* Processing indicator */}
      <ProcessingIndicator
        patientId={processingPatientId}
        visible={!!processingPatientId}
        currentPhase={currentPhase}
      />
    </div>
  );
}

export default App;
