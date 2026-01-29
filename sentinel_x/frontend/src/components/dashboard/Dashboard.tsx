import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { SystemStatus } from './SystemStatus';
import { StartButton } from './StartButton';
import { PriorityStats } from './PriorityStats';
import { HowItWorks } from './HowItWorks';
import type { SystemStatus as SystemStatusType } from '@/types';

interface DashboardProps {
  status: SystemStatusType | null;
  loading: boolean;
  stats: {
    total: number;
    byPriority: Record<number, number>;
  };
  onStart: () => void;
  onStop: () => void;
  onReset: () => void;
}

export function Dashboard({
  status,
  loading,
  stats,
  onStart,
  onStop,
  onReset,
}: DashboardProps) {
  return (
    <div className="container mx-auto py-8 px-4 max-w-6xl">
      {/* Header */}
      <div className="text-center mb-8">
        <h1 className="text-4xl font-bold text-primary mb-2">Sentinel-X</h1>
        <p className="text-xl text-muted-foreground">
          AI-Powered CT Scan Triage System
        </p>
      </div>

      {/* Main controls */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
        {/* Demo Control */}
        <Card className="md:col-span-1">
          <CardHeader>
            <CardTitle>Demo Control</CardTitle>
            <CardDescription>
              Start the simulator to begin processing CT scans
            </CardDescription>
          </CardHeader>
          <CardContent className="flex justify-center">
            <StartButton
              demoStatus={status?.demo_status || 'stopped'}
              loading={loading}
              onStart={onStart}
              onStop={onStop}
              onReset={onReset}
            />
          </CardContent>
        </Card>

        {/* System Status */}
        <Card className="md:col-span-1">
          <CardHeader>
            <CardTitle>System Status</CardTitle>
            <CardDescription>
              Current state of demo components
            </CardDescription>
          </CardHeader>
          <CardContent>
            <SystemStatus status={status} />
          </CardContent>
        </Card>

        {/* Priority Statistics */}
        <Card className="md:col-span-1">
          <CardHeader>
            <CardTitle>Priority Statistics</CardTitle>
            <CardDescription>
              Breakdown of triaged patients by priority
            </CardDescription>
          </CardHeader>
          <CardContent>
            <PriorityStats byPriority={stats.byPriority} total={stats.total} />
          </CardContent>
        </Card>
      </div>

      {/* How It Works */}
      <Card>
        <CardHeader>
          <CardTitle>How It Works</CardTitle>
          <CardDescription>
            Sentinel-X automates CT scan prioritization using AI
          </CardDescription>
        </CardHeader>
        <CardContent>
          <HowItWorks />
        </CardContent>
      </Card>
    </div>
  );
}
