import { useState, useCallback, useEffect } from 'react';
import type { SystemStatus, DemoControlResponse } from '@/types';

const API_BASE = '/api';

export function useDemoControls() {
  const [status, setStatus] = useState<SystemStatus | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchStatus = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE}/demo/status`);
      if (!response.ok) throw new Error('Failed to fetch status');
      const data: SystemStatus = await response.json();
      setStatus(data);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch status');
    }
  }, []);

  const startDemo = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch(`${API_BASE}/demo/start`, { method: 'POST' });
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to start demo');
      }
      const data: DemoControlResponse = await response.json();
      setStatus(data.status);
      return true;
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to start demo');
      return false;
    } finally {
      setLoading(false);
    }
  }, []);

  const stopDemo = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch(`${API_BASE}/demo/stop`, { method: 'POST' });
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to stop demo');
      }
      const data: DemoControlResponse = await response.json();
      setStatus(data.status);
      return true;
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to stop demo');
      return false;
    } finally {
      setLoading(false);
    }
  }, []);

  const resetDemo = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch(`${API_BASE}/demo/reset`, { method: 'POST' });
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to reset demo');
      }
      const data: DemoControlResponse = await response.json();
      setStatus(data.status);
      return true;
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to reset demo');
      return false;
    } finally {
      setLoading(false);
    }
  }, []);

  // Fetch initial status on mount
  useEffect(() => {
    fetchStatus();
  }, [fetchStatus]);

  // Update status from external source (e.g., WebSocket)
  const updateStatus = useCallback((newStatus: SystemStatus) => {
    setStatus(newStatus);
  }, []);

  return {
    status,
    loading,
    error,
    startDemo,
    stopDemo,
    resetDemo,
    fetchStatus,
    updateStatus,
  };
}
