import { useState, useEffect, useRef, useCallback } from 'react';
import { ChevronLeft, ChevronRight, Loader2 } from 'lucide-react';
import { Button } from '@/components/ui/button';

interface CTViewerProps {
  patientId: string;
  keySliceIndex: number;
  totalSlices: number;
  thumbnailBase64?: string;
  getSliceUrl: (patientId: string, sliceIndex: number) => string;
}

const DEBOUNCE_MS = 50;
const CACHE_MAX = 50;
const PREFETCH_OFFSETS = [1, -1, 2, -2, 3, -3];

export function CTViewer({
  patientId,
  keySliceIndex,
  totalSlices,
  thumbnailBase64,
  getSliceUrl,
}: CTViewerProps) {
  // displaySlice updates instantly (for label); fetchSlice updates after debounce (triggers fetch)
  const [displaySlice, setDisplaySlice] = useState(keySliceIndex);
  const [fetchSlice, setFetchSlice] = useState(keySliceIndex);
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // LRU blob cache: Map preserves insertion order for eviction
  const blobCacheRef = useRef<Map<string, string>>(new Map());
  const debounceTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  // Helper: cache key
  const cacheKey = useCallback(
    (slice: number) => `${patientId}:${slice}`,
    [patientId],
  );

  // Helper: add to LRU cache with eviction
  const cacheSet = useCallback(
    (slice: number, url: string) => {
      const cache = blobCacheRef.current;
      const key = cacheKey(slice);
      // Move to end if already present (refresh LRU position)
      if (cache.has(key)) {
        cache.delete(key);
      }
      cache.set(key, url);
      // Evict oldest entries over limit
      while (cache.size > CACHE_MAX) {
        const oldest = cache.keys().next().value!;
        const oldUrl = cache.get(oldest)!;
        URL.revokeObjectURL(oldUrl);
        cache.delete(oldest);
      }
    },
    [cacheKey],
  );

  // Helper: get from cache (returns undefined on miss)
  const cacheGet = useCallback(
    (slice: number): string | undefined => {
      const cache = blobCacheRef.current;
      const key = cacheKey(slice);
      const url = cache.get(key);
      if (url !== undefined) {
        // Move to end (most recently used)
        cache.delete(key);
        cache.set(key, url);
      }
      return url;
    },
    [cacheKey],
  );

  // Prefetch a single slice silently into cache
  const prefetchSlice = useCallback(
    (slice: number) => {
      if (slice < 0 || slice >= totalSlices) return;
      if (blobCacheRef.current.has(cacheKey(slice))) return;

      const url = getSliceUrl(patientId, slice);
      fetch(url)
        .then((res) => {
          if (!res.ok) return;
          return res.blob();
        })
        .then((blob) => {
          if (!blob) return;
          // Check again in case it was cached by another fetch
          if (!blobCacheRef.current.has(cacheKey(slice))) {
            cacheSet(slice, URL.createObjectURL(blob));
          }
        })
        .catch(() => {
          /* prefetch failures are silent */
        });
    },
    [patientId, totalSlices, getSliceUrl, cacheKey, cacheSet],
  );

  // Set both display and fetch slice immediately (for buttons/keyboard)
  const goToSlice = useCallback(
    (slice: number) => {
      const clamped = Math.max(0, Math.min(totalSlices - 1, slice));
      setDisplaySlice(clamped);
      setFetchSlice(clamped);
      // Clear any pending debounce
      if (debounceTimerRef.current) {
        clearTimeout(debounceTimerRef.current);
        debounceTimerRef.current = null;
      }
    },
    [totalSlices],
  );

  // Slider onChange: update display instantly, show cached slices immediately, debounce fetch
  const onSliderChange = useCallback(
    (value: number) => {
      setDisplaySlice(value);

      // Instant display for thumbnail or cached slices
      if (value === keySliceIndex && thumbnailBase64) {
        setImageUrl(`data:image/png;base64,${thumbnailBase64}`);
      } else {
        const key = `${patientId}:${value}`;
        const cached = blobCacheRef.current.get(key);
        if (cached) {
          setImageUrl(cached);
        }
      }

      // Debounce the fetch trigger (for uncached slices + prefetch)
      if (debounceTimerRef.current) {
        clearTimeout(debounceTimerRef.current);
      }
      debounceTimerRef.current = setTimeout(() => {
        setFetchSlice(value);
        debounceTimerRef.current = null;
      }, DEBOUNCE_MS);
    },
    [keySliceIndex, thumbnailBase64, patientId],
  );

  // Fetch effect: watches fetchSlice, uses AbortController
  useEffect(() => {
    const controller = new AbortController();

    // Key slice with thumbnail: use it directly
    if (fetchSlice === keySliceIndex && thumbnailBase64) {
      setImageUrl(`data:image/png;base64,${thumbnailBase64}`);
      setLoading(false);
      setError(null);
      return;
    }

    // Check blob cache first
    const cached = cacheGet(fetchSlice);
    if (cached) {
      setImageUrl(cached);
      setLoading(false);
      setError(null);
      // Still prefetch neighbors
      for (const offset of PREFETCH_OFFSETS) {
        prefetchSlice(fetchSlice + offset);
      }
      return;
    }

    // Fetch from server
    setLoading(true);
    setError(null);

    const url = getSliceUrl(patientId, fetchSlice);
    fetch(url, { signal: controller.signal })
      .then((res) => {
        if (!res.ok) throw new Error('Failed to load slice');
        return res.blob();
      })
      .then((blob) => {
        const blobUrl = URL.createObjectURL(blob);
        cacheSet(fetchSlice, blobUrl);
        setImageUrl(blobUrl);
        setLoading(false);
        // Prefetch adjacent slices
        for (const offset of PREFETCH_OFFSETS) {
          prefetchSlice(fetchSlice + offset);
        }
      })
      .catch((err) => {
        if (err.name === 'AbortError') return; // Expected during fast navigation
        setError('Failed to load CT slice');
        setLoading(false);
        // Fall back to thumbnail
        if (thumbnailBase64) {
          setImageUrl(`data:image/png;base64,${thumbnailBase64}`);
        }
      });

    return () => {
      controller.abort();
    };
  }, [fetchSlice, keySliceIndex, thumbnailBase64, patientId, getSliceUrl, cacheGet, cacheSet, prefetchSlice]);

  // Cleanup all blob URLs on unmount
  useEffect(() => {
    return () => {
      blobCacheRef.current.forEach((url) => URL.revokeObjectURL(url));
      blobCacheRef.current.clear();
    };
  }, []);

  // Clean up debounce timer on unmount
  useEffect(() => {
    return () => {
      if (debounceTimerRef.current) {
        clearTimeout(debounceTimerRef.current);
      }
    };
  }, []);

  // Mouse wheel navigation
  const onWheel = useCallback(
    (e: React.WheelEvent) => {
      e.preventDefault();
      if (e.deltaY > 0) {
        goToSlice(displaySlice + 1);
      } else if (e.deltaY < 0) {
        goToSlice(displaySlice - 1);
      }
    },
    [displaySlice, goToSlice],
  );

  // Keyboard navigation
  const onKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') {
        e.preventDefault();
        goToSlice(displaySlice - 1);
      } else if (e.key === 'ArrowRight' || e.key === 'ArrowDown') {
        e.preventDefault();
        goToSlice(displaySlice + 1);
      }
    },
    [displaySlice, goToSlice],
  );

  return (
    <div
      ref={containerRef}
      className="flex flex-col gap-4 focus:ring-2 focus:ring-primary/50 outline-none rounded-lg"
      tabIndex={0}
      onKeyDown={onKeyDown}
    >
      {/* Image container */}
      <div className="relative bg-black rounded-lg overflow-hidden aspect-square flex items-center justify-center" onWheel={onWheel}>
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
            alt={`CT slice ${displaySlice}`}
            className="max-w-full max-h-full object-contain"
          />
        )}

        {/* Slice info overlay */}
        <div className="absolute bottom-2 left-2 bg-black/70 text-white text-xs px-2 py-1 rounded">
          Slice {displaySlice + 1} / {totalSlices}
        </div>

        {displaySlice === keySliceIndex && (
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
          onClick={() => goToSlice(displaySlice - 1)}
          disabled={displaySlice === 0}
        >
          <ChevronLeft className="h-4 w-4" />
          Prev
        </Button>

        <Button
          variant="outline"
          size="sm"
          onClick={() => goToSlice(keySliceIndex)}
          disabled={displaySlice === keySliceIndex}
        >
          Go to Key Slice
        </Button>

        <Button
          variant="outline"
          size="sm"
          onClick={() => goToSlice(displaySlice + 1)}
          disabled={displaySlice === totalSlices - 1}
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
        value={displaySlice}
        onChange={(e) => onSliderChange(parseInt(e.target.value))}
        className="w-full"
      />
    </div>
  );
}
