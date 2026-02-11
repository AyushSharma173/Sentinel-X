import { Play, Square, RotateCcw, Loader2, CheckCircle } from 'lucide-react';
import { Button } from '@/components/ui/button';
import type { DemoStatus } from '@/types';

interface StartButtonProps {
  demoStatus: DemoStatus;
  loading: boolean;
  onStart: () => void;
  onStop: () => void;
  onReset: () => void;
}

export function StartButton({
  demoStatus,
  loading,
  onStart,
  onStop,
  onReset,
}: StartButtonProps) {
  const isRunning = demoStatus === 'running';
  const isStopped = demoStatus === 'stopped';
  const isCompleted = demoStatus === 'completed';
  const isTransitioning = demoStatus === 'starting' || demoStatus === 'stopping';

  return (
    <div className="flex flex-col items-center gap-4">
      {isCompleted ? (
        <Button
          size="xl"
          disabled
          className="gap-2 min-w-[200px] bg-routine hover:bg-routine text-white"
        >
          <CheckCircle className="h-6 w-6" />
          Demo Complete
        </Button>
      ) : isStopped ? (
        <Button
          size="xl"
          onClick={onStart}
          disabled={loading}
          className="gap-2 min-w-[200px]"
        >
          {loading ? (
            <Loader2 className="h-6 w-6 animate-spin" />
          ) : (
            <Play className="h-6 w-6" />
          )}
          Start Demo
        </Button>
      ) : isRunning ? (
        <Button
          size="xl"
          variant="destructive"
          onClick={onStop}
          disabled={loading}
          className="gap-2 min-w-[200px]"
        >
          {loading ? (
            <Loader2 className="h-6 w-6 animate-spin" />
          ) : (
            <Square className="h-6 w-6" />
          )}
          Stop Demo
        </Button>
      ) : (
        <Button
          size="xl"
          disabled
          className="gap-2 min-w-[200px]"
        >
          <Loader2 className="h-6 w-6 animate-spin" />
          {demoStatus === 'starting' ? 'Starting...' : 'Stopping...'}
        </Button>
      )}

      {(isStopped || isRunning || isCompleted) && (
        <Button
          variant="outline"
          size="sm"
          onClick={onReset}
          disabled={loading || isTransitioning}
          className="gap-2"
        >
          <RotateCcw className="h-4 w-4" />
          Reset Demo
        </Button>
      )}
    </div>
  );
}
