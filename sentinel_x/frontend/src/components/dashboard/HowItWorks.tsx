import { Scan, Brain, ListOrdered, AlertTriangle } from 'lucide-react';

const steps = [
  {
    icon: Scan,
    title: 'CT Scan Arrival',
    description: 'New CT scans arrive in the inbox with patient reports',
  },
  {
    icon: Brain,
    title: 'AI Analysis',
    description: 'MedGemma analyzes images combined with EHR context',
  },
  {
    icon: AlertTriangle,
    title: 'Priority Assessment',
    description: 'Acute pathology and risk factors determine priority',
  },
  {
    icon: ListOrdered,
    title: 'Worklist Update',
    description: 'Cases are sorted by priority for radiologist review',
  },
];

export function HowItWorks() {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
      {steps.map((step, index) => (
        <div
          key={index}
          className="flex flex-col items-center text-center p-4 rounded-lg bg-muted/50"
        >
          <div className="mb-3 p-3 rounded-full bg-primary/10">
            <step.icon className="h-6 w-6 text-primary" />
          </div>
          <h4 className="font-medium mb-1">{step.title}</h4>
          <p className="text-sm text-muted-foreground">{step.description}</p>
        </div>
      ))}
    </div>
  );
}
