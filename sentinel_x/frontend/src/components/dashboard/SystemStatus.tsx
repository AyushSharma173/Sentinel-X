import { CheckCircle, XCircle, Loader2, AlertCircle } from 'lucide-react';
import type { SystemStatus as SystemStatusType } from '@/types';

interface SystemStatusProps {
  status: SystemStatusType | null;
}

interface StatusIndicatorProps {
  label: string;
  active: boolean;
  loading?: boolean;
}

function StatusIndicator({ label, active, loading }: StatusIndicatorProps) {
  return (
    <div className="flex items-center gap-2">
      {loading ? (
        <Loader2 className="h-4 w-4 text-primary animate-spin" />
      ) : active ? (
        <CheckCircle className="h-4 w-4 text-routine" />
      ) : (
        <XCircle className="h-4 w-4 text-gray-400" />
      )}
      <span className={active ? 'text-foreground' : 'text-muted-foreground'}>
        {label}
      </span>
    </div>
  );
}

export function SystemStatus({ status }: SystemStatusProps) {
  if (!status) {
    return (
      <div className="flex items-center gap-2 text-muted-foreground">
        <Loader2 className="h-4 w-4 animate-spin" />
        <span>Loading status...</span>
      </div>
    );
  }

  const isStarting = status.demo_status === 'starting';
  const isStopping = status.demo_status === 'stopping';

  return (
    <div className="flex flex-col gap-4">
      <div className="grid grid-cols-2 gap-4">
        <StatusIndicator
          label="Simulator"
          active={status.simulator_running}
          loading={isStarting && !status.simulator_running}
        />
        <StatusIndicator
          label="Triage Agent"
          active={status.agent_running}
          loading={isStarting && !status.agent_running}
        />
        <StatusIndicator
          label="MedGemma Model"
          active={status.model_loaded}
          loading={isStarting && !status.model_loaded}
        />
        <div className="flex items-center gap-2">
          {status.demo_status === 'running' ? (
            <AlertCircle className="h-4 w-4 text-primary" />
          ) : (
            <div className="h-4 w-4 rounded-full bg-gray-200" />
          )}
          <span className={status.demo_status === 'running' ? 'text-primary font-medium' : 'text-muted-foreground'}>
            {status.demo_status === 'running' ? 'Demo Active' : 'Demo Inactive'}
          </span>
        </div>
      </div>

      {(isStarting || isStopping) && (
        <div className="flex items-center gap-2 text-sm text-muted-foreground">
          <Loader2 className="h-4 w-4 animate-spin" />
          <span>{isStarting ? 'Starting demo...' : 'Stopping demo...'}</span>
        </div>
      )}
    </div>
  );
}
