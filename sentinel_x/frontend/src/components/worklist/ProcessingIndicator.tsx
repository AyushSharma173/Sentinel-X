import { Loader2, User } from 'lucide-react';

interface ProcessingIndicatorProps {
  patientId: string | null;
  visible: boolean;
}

export function ProcessingIndicator({ patientId, visible }: ProcessingIndicatorProps) {
  if (!visible || !patientId) return null;

  return (
    <div className="fixed bottom-4 right-4 bg-primary text-white px-4 py-3 rounded-lg shadow-lg flex items-center gap-3 animate-fade-in z-50">
      <Loader2 className="h-5 w-5 animate-spin" />
      <div>
        <div className="flex items-center gap-2">
          <User className="h-4 w-4" />
          <span className="font-medium">{patientId}</span>
        </div>
        <span className="text-sm text-white/80">Processing...</span>
      </div>
    </div>
  );
}
