import { Eye, ArrowRightLeft, Brain, User } from 'lucide-react';

interface ProcessingIndicatorProps {
  patientId: string | null;
  visible: boolean;
  currentPhase?: string | null;
}

const PHASE_CONFIG: Record<string, { icon: typeof Eye; label: string; step: number }> = {
  phase1: { icon: Eye, label: 'Visual Analysis', step: 0 },
  model_swap: { icon: ArrowRightLeft, label: 'Model Swap', step: 1 },
  phase2: { icon: Brain, label: 'Clinical Reasoning', step: 2 },
};

function ProgressDots({ currentStep }: { currentStep: number }) {
  return (
    <div className="flex items-center gap-1.5 mt-1">
      {[0, 1, 2].map((step) => (
        <div
          key={step}
          className={`h-1.5 w-1.5 rounded-full transition-all ${
            step < currentStep
              ? 'bg-white'
              : step === currentStep
                ? 'bg-white animate-pulse'
                : 'bg-white/30'
          }`}
        />
      ))}
    </div>
  );
}

export function ProcessingIndicator({ patientId, visible, currentPhase }: ProcessingIndicatorProps) {
  if (!visible || !patientId) return null;

  const phase = currentPhase ? PHASE_CONFIG[currentPhase] : null;
  const PhaseIcon = phase?.icon;

  return (
    <div className="fixed bottom-4 right-4 bg-primary text-white px-4 py-3 rounded-lg shadow-lg flex items-center gap-3 animate-fade-in z-50">
      {PhaseIcon ? (
        <PhaseIcon className="h-5 w-5 animate-pulse" />
      ) : (
        <div className="h-5 w-5 animate-spin rounded-full border-2 border-white border-t-transparent" />
      )}
      <div>
        <div className="flex items-center gap-2">
          <User className="h-4 w-4" />
          <span className="font-medium">{patientId}</span>
        </div>
        <span className="text-sm text-white/80">
          {phase ? phase.label : 'Processing...'}
        </span>
        {phase && <ProgressDots currentStep={phase.step} />}
      </div>
    </div>
  );
}
