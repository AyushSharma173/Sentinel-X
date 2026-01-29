import { Button } from '@/components/ui/button';
import { cn } from '@/lib/utils';

interface PriorityFilterProps {
  selected: number | null;
  onChange: (priority: number | null) => void;
  counts: Record<number, number>;
}

const filters = [
  { value: null, label: 'All', colorClass: 'bg-primary' },
  { value: 1, label: 'Critical', colorClass: 'bg-critical' },
  { value: 2, label: 'High Risk', colorClass: 'bg-highrisk' },
  { value: 3, label: 'Routine', colorClass: 'bg-routine' },
];

export function PriorityFilter({ selected, onChange, counts }: PriorityFilterProps) {
  const totalCount = Object.values(counts).reduce((a, b) => a + b, 0);

  return (
    <div className="flex gap-2 flex-wrap">
      {filters.map(({ value, label, colorClass }) => {
        const count = value === null ? totalCount : (counts[value] || 0);
        const isSelected = selected === value;

        return (
          <Button
            key={label}
            variant={isSelected ? 'default' : 'outline'}
            size="sm"
            onClick={() => onChange(value)}
            className={cn(
              'gap-2',
              isSelected && value !== null && colorClass,
              isSelected && 'text-white'
            )}
          >
            {label}
            <span className={cn(
              'inline-flex items-center justify-center px-1.5 py-0.5 text-xs rounded-full',
              isSelected ? 'bg-white/20' : 'bg-muted'
            )}>
              {count}
            </span>
          </Button>
        );
      })}
    </div>
  );
}
