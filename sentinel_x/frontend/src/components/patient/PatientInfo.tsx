import { User, AlertTriangle, Pill, FileText } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import type { PatientFHIRContext } from '@/types';

interface PatientInfoProps {
  context: PatientFHIRContext;
}

export function PatientInfo({ context }: PatientInfoProps) {
  const { demographics, conditions, medications, risk_factors, findings, impressions } = context;

  return (
    <div className="space-y-4">
      {/* Demographics */}
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-base flex items-center gap-2">
            <User className="h-4 w-4 text-primary" />
            Demographics
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 gap-4 text-sm">
            <div>
              <span className="text-muted-foreground">Patient ID:</span>
              <span className="ml-2 font-medium">{demographics.patient_id}</span>
            </div>
            {demographics.age && (
              <div>
                <span className="text-muted-foreground">Age:</span>
                <span className="ml-2 font-medium">{demographics.age} years</span>
              </div>
            )}
            {demographics.gender && (
              <div>
                <span className="text-muted-foreground">Gender:</span>
                <span className="ml-2 font-medium capitalize">{demographics.gender}</span>
              </div>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Conditions & Risk Factors */}
      {(conditions.length > 0 || risk_factors.length > 0) && (
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-base flex items-center gap-2">
              <AlertTriangle className="h-4 w-4 text-primary" />
              Medical History
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            {conditions.length > 0 && (
              <div>
                <p className="text-xs text-muted-foreground mb-2">Conditions:</p>
                <div className="flex flex-wrap gap-2">
                  {conditions.map((condition, index) => (
                    <Badge
                      key={index}
                      variant={condition.is_risk_factor ? 'highrisk' : 'secondary'}
                    >
                      {condition.name}
                      {condition.is_risk_factor && ' (Risk)'}
                    </Badge>
                  ))}
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {/* Medications */}
      {medications.length > 0 && (
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-base flex items-center gap-2">
              <Pill className="h-4 w-4 text-primary" />
              Current Medications
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex flex-wrap gap-2">
              {medications.map((medication, index) => (
                <Badge key={index} variant="outline">
                  {medication}
                </Badge>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Radiology Report */}
      {(findings || impressions) && (
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-base flex items-center gap-2">
              <FileText className="h-4 w-4 text-primary" />
              Radiology Report
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            {findings && (
              <div>
                <p className="text-xs font-medium text-muted-foreground mb-1">Findings:</p>
                <p className="text-sm">{findings}</p>
              </div>
            )}
            {impressions && (
              <div>
                <p className="text-xs font-medium text-muted-foreground mb-1">Impressions:</p>
                <p className="text-sm">{impressions}</p>
              </div>
            )}
          </CardContent>
        </Card>
      )}
    </div>
  );
}
