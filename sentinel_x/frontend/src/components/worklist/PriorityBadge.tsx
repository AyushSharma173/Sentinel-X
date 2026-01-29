import { Badge } from '@/components/ui/badge';
import { cn } from '@/lib/utils';

interface PriorityBadgeProps {
  level: number;
  name: string;
  animate?: boolean;
  size?: 'sm' | 'md' | 'lg';
}

export function PriorityBadge({
  level,
  name,
  animate = false,
  size = 'md',
}: PriorityBadgeProps) {
  const variant = level === 1 ? 'critical' : level === 2 ? 'highrisk' : 'routine';

  const sizeClasses = {
    sm: 'text-xs px-2 py-0.5',
    md: 'text-sm px-2.5 py-0.5',
    lg: 'text-base px-3 py-1',
  };

  return (
    <Badge
      variant={variant}
      className={cn(
        sizeClasses[size],
        animate && level === 1 && 'animate-pulse-priority'
      )}
    >
      P{level}: {name}
    </Badge>
  );
}
