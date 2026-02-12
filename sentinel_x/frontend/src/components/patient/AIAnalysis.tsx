import ReactMarkdown from 'react-markdown';
import { Brain, Eye } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

interface AIAnalysisProps {
  phase1Raw?: string;
  reasoning?: string;
}

export function AIAnalysis({
  phase1Raw,
  reasoning,
}: AIAnalysisProps) {
  return (
    <div className="space-y-4">
      {/* Visual Findings */}
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-base flex items-center gap-2">
            <Eye className="h-4 w-4 text-primary" />
            Visual Findings
          </CardTitle>
        </CardHeader>
        <CardContent>
          {phase1Raw ? (
            <div className="text-sm text-muted-foreground [&_strong]:text-foreground [&_strong]:font-semibold [&_ul]:list-disc [&_ul]:pl-5 [&_ul]:space-y-1 [&_ol]:list-decimal [&_ol]:pl-5 [&_ol]:space-y-1 [&_p]:mb-2 [&_hr]:my-3 [&_hr]:border-border">
              <ReactMarkdown>{phase1Raw}</ReactMarkdown>
            </div>
          ) : (
            <p className="text-sm text-muted-foreground">No visual findings available</p>
          )}
        </CardContent>
      </Card>

      {/* Clinical Reasoning */}
      {reasoning && (
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-base flex items-center gap-2">
              <Brain className="h-4 w-4 text-primary" />
              Clinical Reasoning
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-sm text-muted-foreground [&_strong]:text-foreground [&_strong]:font-semibold [&_ul]:list-disc [&_ul]:pl-5 [&_ul]:space-y-1 [&_ol]:list-decimal [&_ol]:pl-5 [&_ol]:space-y-1 [&_p]:mb-2 [&_hr]:my-3 [&_hr]:border-border">
              <ReactMarkdown>{reasoning}</ReactMarkdown>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
