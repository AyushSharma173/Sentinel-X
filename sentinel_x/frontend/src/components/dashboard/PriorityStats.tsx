import { Badge } from '@/components/ui/badge';

interface PriorityStatsProps {
  byPriority: Record<number, number>;
  total: number;
}

export function PriorityStats({ byPriority, total }: PriorityStatsProps) {
  const p1 = byPriority[1] || 0;
  const p2 = byPriority[2] || 0;
  const p3 = byPriority[3] || 0;

  return (
    <div className="flex flex-col gap-4">
      <div className="text-center">
        <div className="text-4xl font-bold text-primary">{total}</div>
        <div className="text-sm text-muted-foreground">Patients Triaged</div>
      </div>

      <div className="flex justify-center gap-6">
        <div className="flex flex-col items-center gap-1">
          <Badge variant="critical" className="text-lg px-3 py-1">
            {p1}
          </Badge>
          <span className="text-xs text-muted-foreground">Critical</span>
        </div>

        <div className="flex flex-col items-center gap-1">
          <Badge variant="highrisk" className="text-lg px-3 py-1">
            {p2}
          </Badge>
          <span className="text-xs text-muted-foreground">High Risk</span>
        </div>

        <div className="flex flex-col items-center gap-1">
          <Badge variant="routine" className="text-lg px-3 py-1">
            {p3}
          </Badge>
          <span className="text-xs text-muted-foreground">Routine</span>
        </div>
      </div>
    </div>
  );
}
