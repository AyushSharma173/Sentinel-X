import { Brain, Eye, FileText } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

interface AIAnalysisProps {
  visualFindings: string;
  conditionsConsidered: string[];
  rationale: string;
  headline?: string;
  reasoning?: string;
}

export function AIAnalysis({
  visualFindings,
  conditionsConsidered,
  rationale,
  headline,
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
          <p className="text-sm text-muted-foreground">{visualFindings || 'No visual findings available'}</p>
        </CardContent>
      </Card>

      {/* Conditions Considered */}
      {conditionsConsidered.length > 0 && (
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-base flex items-center gap-2">
              <Brain className="h-4 w-4 text-primary" />
              Conditions Considered
            </CardTitle>
          </CardHeader>
          <CardContent>
            <ul className="text-sm text-muted-foreground space-y-1">
              {conditionsConsidered.map((condition, index) => (
                <li key={index} className="flex items-start gap-2">
                  <span className="text-primary">•</span>
                  {condition}
                </li>
              ))}
            </ul>
          </CardContent>
        </Card>
      )}

      {/* Priority Rationale */}
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-base flex items-center gap-2">
            <FileText className="h-4 w-4 text-primary" />
            Priority Rationale
          </CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-muted-foreground">{rationale || 'No rationale available'}</p>
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
            <p className="text-sm text-muted-foreground whitespace-pre-line">{reasoning}</p>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
