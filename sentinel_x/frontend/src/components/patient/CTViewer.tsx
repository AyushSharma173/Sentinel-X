import { useState, useEffect } from 'react';
import { ChevronLeft, ChevronRight, Loader2 } from 'lucide-react';
import { Button } from '@/components/ui/button';

interface CTViewerProps {
  patientId: string;
  keySliceIndex: number;
  totalSlices: number;
  thumbnailBase64?: string;
  getSliceUrl: (patientId: string, sliceIndex: number) => string;
}

export function CTViewer({
  patientId,
  keySliceIndex,
  totalSlices,
  thumbnailBase64,
  getSliceUrl,
}: CTViewerProps) {
  const [currentSlice, setCurrentSlice] = useState(keySliceIndex);
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Load slice when current slice changes
  useEffect(() => {
    const loadSlice = async () => {
      // For the key slice, use the thumbnail if available
      if (currentSlice === keySliceIndex && thumbnailBase64) {
        setImageUrl(`data:image/png;base64,${thumbnailBase64}`);
        return;
      }

      setLoading(true);
      setError(null);

      try {
        const url = getSliceUrl(patientId, currentSlice);
        // Verify the image can load
        const response = await fetch(url);
        if (!response.ok) throw new Error('Failed to load slice');
        setImageUrl(url);
      } catch (err) {
        setError('Failed to load CT slice');
        // Fall back to thumbnail
        if (thumbnailBase64) {
          setImageUrl(`data:image/png;base64,${thumbnailBase64}`);
        }
      } finally {
        setLoading(false);
      }
    };

    loadSlice();
  }, [currentSlice, keySliceIndex, thumbnailBase64, patientId, getSliceUrl]);

  const goToPrevSlice = () => {
    setCurrentSlice((prev) => Math.max(0, prev - 1));
  };

  const goToNextSlice = () => {
    setCurrentSlice((prev) => Math.min(totalSlices - 1, prev + 1));
  };

  const goToKeySlice = () => {
    setCurrentSlice(keySliceIndex);
  };

  return (
    <div className="flex flex-col gap-4">
      {/* Image container */}
      <div className="relative bg-black rounded-lg overflow-hidden aspect-square flex items-center justify-center">
        {loading && (
          <div className="absolute inset-0 flex items-center justify-center bg-black/50">
            <Loader2 className="h-8 w-8 animate-spin text-white" />
          </div>
        )}

        {error && !imageUrl && (
          <div className="text-white text-center p-4">
            <p>{error}</p>
          </div>
        )}

        {imageUrl && (
          <img
            src={imageUrl}
            alt={`CT slice ${currentSlice}`}
            className="max-w-full max-h-full object-contain"
          />
        )}

        {/* Slice info overlay */}
        <div className="absolute bottom-2 left-2 bg-black/70 text-white text-xs px-2 py-1 rounded">
          Slice {currentSlice + 1} / {totalSlices}
        </div>

        {currentSlice === keySliceIndex && (
          <div className="absolute top-2 right-2 bg-primary text-white text-xs px-2 py-1 rounded">
            Key Finding
          </div>
        )}
      </div>

      {/* Navigation */}
      <div className="flex items-center justify-between gap-2">
        <Button
          variant="outline"
          size="sm"
          onClick={goToPrevSlice}
          disabled={currentSlice === 0 || loading}
        >
          <ChevronLeft className="h-4 w-4" />
          Prev
        </Button>

        <Button
          variant="outline"
          size="sm"
          onClick={goToKeySlice}
          disabled={currentSlice === keySliceIndex}
        >
          Go to Key Slice
        </Button>

        <Button
          variant="outline"
          size="sm"
          onClick={goToNextSlice}
          disabled={currentSlice === totalSlices - 1 || loading}
        >
          Next
          <ChevronRight className="h-4 w-4" />
        </Button>
      </div>

      {/* Slider */}
      <input
        type="range"
        min={0}
        max={totalSlices - 1}
        value={currentSlice}
        onChange={(e) => setCurrentSlice(parseInt(e.target.value))}
        className="w-full"
      />
    </div>
  );
}
