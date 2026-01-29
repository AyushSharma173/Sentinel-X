import { Loader2, Inbox } from 'lucide-react';
import { WorklistRow } from './WorklistRow';
import { PriorityFilter } from './PriorityFilter';
import type { WorklistEntry } from '@/types';

interface WorklistTableProps {
  entries: WorklistEntry[];
  loading: boolean;
  priorityFilter: number | null;
  onFilterChange: (priority: number | null) => void;
  onRowClick: (patientId: string) => void;
  counts: Record<number, number>;
  newPatientIds?: Set<string>;
}

export function WorklistTable({
  entries,
  loading,
  priorityFilter,
  onFilterChange,
  onRowClick,
  counts,
  newPatientIds = new Set(),
}: WorklistTableProps) {
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
      ) : entries.length === 0 ? (
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
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
