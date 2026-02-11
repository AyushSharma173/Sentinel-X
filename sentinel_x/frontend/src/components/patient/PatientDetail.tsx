import { X, Loader2, Clock } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Badge } from '@/components/ui/badge';
import { PriorityBadge } from '@/components/worklist/PriorityBadge';
import { CTViewer } from './CTViewer';
import { AIAnalysis } from './AIAnalysis';
import { PatientInfo } from './PatientInfo';
import type { TriageResult, PatientFHIRContext, VolumeInfo } from '@/types';

interface PatientDetailProps {
  triageResult: TriageResult | null;
  fhirContext: PatientFHIRContext | null;
  volumeInfo: VolumeInfo | null;
  loading: boolean;
  error: string | null;
  onClose: () => void;
  getSliceUrl: (patientId: string, sliceIndex: number) => string;
  selectedPatientId?: string | null;
}

export function PatientDetail({
  triageResult,
  fhirContext,
  volumeInfo,
  loading,
  error,
  onClose,
  getSliceUrl,
  selectedPatientId,
}: PatientDetailProps) {
  const patientId = triageResult?.patient_id || selectedPatientId;
  const hasContent = triageResult || fhirContext || volumeInfo;

  return (
    <div className="fixed inset-y-0 right-0 w-full max-w-2xl bg-white border-l shadow-xl z-50 animate-slide-in-right flex flex-col">
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b">
        <div className="flex items-center gap-4">
          <Button variant="ghost" size="icon" onClick={onClose}>
            <X className="h-5 w-5" />
          </Button>
          {triageResult ? (
            <>
              <h2 className="text-xl font-semibold">{triageResult.patient_id}</h2>
              <PriorityBadge
                level={triageResult.priority_level}
                name={triageResult.priority_name}
                size="lg"
              />
            </>
          ) : patientId ? (
            <>
              <h2 className="text-xl font-semibold">{patientId}</h2>
              <Badge variant="outline">QUEUED FOR TRIAGE</Badge>
            </>
          ) : null}
        </div>
      </div>

      {/* Content */}
      <ScrollArea className="flex-1">
        {loading ? (
          <div className="flex items-center justify-center py-12">
            <Loader2 className="h-8 w-8 animate-spin text-primary" />
          </div>
        ) : error ? (
          <div className="p-6 text-center text-destructive">
            <p>{error}</p>
            <Button variant="outline" className="mt-4" onClick={onClose}>
              Close
            </Button>
          </div>
        ) : hasContent ? (
          <div className="p-6">
            <Tabs defaultValue="imaging" className="w-full">
              <TabsList className="grid w-full grid-cols-3">
                <TabsTrigger value="imaging">CT Imaging</TabsTrigger>
                <TabsTrigger value="analysis">AI Analysis</TabsTrigger>
                <TabsTrigger value="patient">Patient Info</TabsTrigger>
              </TabsList>

              <TabsContent value="imaging" className="mt-4">
                {triageResult ? (
                  <CTViewer
                    patientId={triageResult.patient_id}
                    keySliceIndex={triageResult.key_slice_index}
                    totalSlices={volumeInfo?.total_slices || 85}
                    thumbnailBase64={triageResult.key_slice_thumbnail}
                    getSliceUrl={getSliceUrl}
                  />
                ) : patientId && volumeInfo ? (
                  <CTViewer
                    patientId={patientId}
                    keySliceIndex={Math.floor(volumeInfo.total_slices / 2)}
                    totalSlices={volumeInfo.total_slices}
                    thumbnailBase64=""
                    getSliceUrl={getSliceUrl}
                  />
                ) : (
                  <p className="text-muted-foreground text-center py-4">
                    CT imaging not available yet
                  </p>
                )}
              </TabsContent>

              <TabsContent value="analysis" className="mt-4">
                {triageResult ? (
                  <AIAnalysis
                    visualFindings={triageResult.visual_findings}
                    conditionsConsidered={triageResult.conditions_considered}
                    rationale={triageResult.rationale}
                    headline={triageResult.headline}
                    reasoning={triageResult.reasoning}
                  />
                ) : (
                  <div className="flex flex-col items-center justify-center py-12 text-muted-foreground">
                    <Clock className="h-12 w-12 mb-4" />
                    <p className="text-lg font-medium">Queued for Triage Assessment</p>
                    <p className="text-sm mt-2 text-center max-w-sm">
                      AI analysis will appear here once the triage system processes this patient.
                    </p>
                  </div>
                )}
              </TabsContent>

              <TabsContent value="patient" className="mt-4">
                {fhirContext ? (
                  <PatientInfo context={fhirContext} />
                ) : (
                  <p className="text-muted-foreground text-center py-4">
                    Patient information not available
                  </p>
                )}
              </TabsContent>
            </Tabs>

            {/* Summary section */}
            <div className="mt-6 p-4 bg-muted/50 rounded-lg">
              <h3 className="font-medium mb-2">Findings Summary</h3>
              <p className="text-sm text-muted-foreground">
                {triageResult
                  ? (triageResult.headline || triageResult.findings_summary)
                  : 'Awaiting triage assessment'}
              </p>
            </div>
          </div>
        ) : null}
      </ScrollArea>
    </div>
  );
}
