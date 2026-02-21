import { useState, useCallback, useEffect } from 'react';
import type { WorklistEntry, WorklistResponse } from '@/types';

const API_BASE = '/api';

export function useWorklist() {
  const [entries, setEntries] = useState<WorklistEntry[]>([]);
  const [stats, setStats] = useState<{
    total: number;
    byPriority: Record<number, number>;
    priorityNames: Record<number, string>;
  }>({
    total: 0,
    byPriority: {},
    priorityNames: {},
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [priorityFilter, setPriorityFilter] = useState<number | null>(null);

  const fetchWorklist = useCallback(async (priority?: number | null) => {
    setLoading(true);
    setError(null);
    try {
      const url = priority != null
        ? `${API_BASE}/worklist?priority=${priority}`
        : `${API_BASE}/worklist`;

      const response = await fetch(url, { cache: 'no-store' });
      if (!response.ok) throw new Error('Failed to fetch worklist');

      const data: WorklistResponse = await response.json();
      setEntries(data.entries);
      setStats({
        total: data.total,
        byPriority: data.by_priority,
        priorityNames: data.priority_names,
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch worklist');
    } finally {
      setLoading(false);
    }
  }, []);

  // Fetch on mount and when filter changes
  useEffect(() => {
    fetchWorklist(priorityFilter);
  }, [fetchWorklist, priorityFilter]);

  const refresh = useCallback(() => {
    fetchWorklist(priorityFilter);
  }, [fetchWorklist, priorityFilter]);

  const setFilter = useCallback((priority: number | null) => {
    setPriorityFilter(priority);
  }, []);

  return {
    entries,
    stats,
    loading,
    error,
    priorityFilter,
    setFilter,
    refresh,
  };
}
