import { ChevronRight } from 'lucide-react';
import { PriorityBadge } from './PriorityBadge';
import { formatTimeAgo } from '@/lib/utils';
import type { WorklistEntry } from '@/types';

interface WorklistRowProps {
  entry: WorklistEntry;
  onClick: () => void;
  isNew?: boolean;
}

export function WorklistRow({ entry, onClick, isNew = false }: WorklistRowProps) {
  return (
    <tr
      onClick={onClick}
      className={`
        border-b cursor-pointer transition-colors hover:bg-muted/50
        ${isNew ? 'highlight-new' : ''}
      `}
    >
      {/* Priority */}
      <td className="py-4 px-4">
        <PriorityBadge
          level={entry.priority_level}
          name={entry.priority_name}
          animate={isNew && entry.priority_level === 1}
        />
      </td>

      {/* Patient ID */}
      <td className="py-4 px-4">
        <span className="font-medium">{entry.patient_id}</span>
      </td>

      {/* Findings Summary */}
      <td className="py-4 px-4 max-w-md">
        <p className="text-sm text-muted-foreground truncate">
          {entry.findings_summary}
        </p>
      </td>

      {/* Time */}
      <td className="py-4 px-4 text-sm text-muted-foreground">
        {formatTimeAgo(entry.processed_at)}
      </td>

      {/* Action */}
      <td className="py-4 px-4">
        <ChevronRight className="h-5 w-5 text-muted-foreground" />
      </td>
    </tr>
  );
}
