import { Loader2, Inbox, ChevronRight } from 'lucide-react';
import { WorklistRow } from './WorklistRow';
import { PriorityFilter } from './PriorityFilter';
import { Badge } from '@/components/ui/badge';
import { formatTimeAgo } from '@/lib/utils';
import type { WorklistEntry, QueuedPatient } from '@/types';

const PHASE_LABELS: Record<string, string> = {
  phase1: 'Visual Analysis...',
  model_swap: 'Model Swap...',
  phase2: 'Clinical Reasoning...',
};

interface WorklistTableProps {
  entries: WorklistEntry[];
  loading: boolean;
  priorityFilter: number | null;
  onFilterChange: (priority: number | null) => void;
  onRowClick: (patientId: string) => void;
  counts: Record<number, number>;
  newPatientIds?: Set<string>;
  queuedPatients?: QueuedPatient[];
}

function QueuedPatientRow({
  patient,
  onClick,
}: {
  patient: QueuedPatient;
  onClick: () => void;
}) {
  const isAnalyzing = patient.status === 'analyzing';

  return (
    <tr
      onClick={onClick}
      className="border-b cursor-pointer transition-colors hover:bg-muted/50"
    >
      {/* Status badge */}
      <td className="py-4 px-4">
        {isAnalyzing ? (
          <Badge variant="default" className="animate-pulse">
            ANALYZING
          </Badge>
        ) : (
          <Badge variant="outline">QUEUED</Badge>
        )}
      </td>

      {/* Patient ID */}
      <td className="py-4 px-4">
        <span className="font-medium">{patient.patient_id}</span>
      </td>

      {/* Status / Phase */}
      <td className="py-4 px-4 max-w-md">
        {isAnalyzing && patient.phase ? (
          <div className="flex items-center gap-2 text-sm text-primary">
            <Loader2 className="h-4 w-4 animate-spin" />
            <span>{PHASE_LABELS[patient.phase] || 'Processing...'}</span>
          </div>
        ) : (
          <p className="text-sm text-muted-foreground italic">
            Awaiting triage assessment
          </p>
        )}
      </td>

      {/* Time */}
      <td className="py-4 px-4 text-sm text-muted-foreground">
        {formatTimeAgo(patient.arrived_at)}
      </td>

      {/* Action */}
      <td className="py-4 px-4">
        <ChevronRight className="h-5 w-5 text-muted-foreground" />
      </td>
    </tr>
  );
}

export function WorklistTable({
  entries,
  loading,
  priorityFilter,
  onFilterChange,
  onRowClick,
  counts,
  newPatientIds = new Set(),
  queuedPatients = [],
}: WorklistTableProps) {
  // Filter queued patients that aren't already in the processed entries
  const pendingQueued = queuedPatients.filter(
    (qp) => !entries.some((e) => e.patient_id === qp.patient_id)
  );

  const hasContent = entries.length > 0 || pendingQueued.length > 0;

  return (
    <div className="bg-white rounded-lg border shadow-sm">
      {/* Header */}
      <div className="p-4 border-b flex items-center justify-between flex-wrap gap-4">
        <h2 className="text-xl font-semibold">Triage Worklist</h2>
        <PriorityFilter
          selected={priorityFilter}
          onChange={onFilterChange}
          counts={counts}
        />
      </div>

      {/* Table */}
      {loading ? (
        <div className="flex items-center justify-center py-12">
          <Loader2 className="h-8 w-8 animate-spin text-primary" />
        </div>
      ) : !hasContent ? (
        <div className="flex flex-col items-center justify-center py-12 text-muted-foreground">
          <Inbox className="h-12 w-12 mb-4" />
          <p className="text-lg">No patients in worklist</p>
          <p className="text-sm">Start the demo to begin processing CT scans</p>
        </div>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead className="bg-muted/50">
              <tr>
                <th className="py-3 px-4 text-left text-sm font-medium text-muted-foreground">
                  Priority
                </th>
                <th className="py-3 px-4 text-left text-sm font-medium text-muted-foreground">
                  Patient ID
                </th>
                <th className="py-3 px-4 text-left text-sm font-medium text-muted-foreground">
                  Findings
                </th>
                <th className="py-3 px-4 text-left text-sm font-medium text-muted-foreground">
                  Time
                </th>
                <th className="py-3 px-4 w-10"></th>
              </tr>
            </thead>
            <tbody>
              {entries.map((entry) => (
                <WorklistRow
                  key={entry.patient_id}
                  entry={entry}
                  onClick={() => onRowClick(entry.patient_id)}
                  isNew={newPatientIds.has(entry.patient_id)}
                />
              ))}
              {pendingQueued.map((qp) => (
                <QueuedPatientRow
                  key={qp.patient_id}
                  patient={qp}
                  onClick={() => onRowClick(qp.patient_id)}
                />
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
