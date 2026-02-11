import { useEffect, useRef, useCallback, useState } from 'react';
import type { WSEvent, WSEventType } from '@/types';

type EventCallback = (data: Record<string, unknown>) => void;

interface UseTriageWebSocketOptions {
  onDemoStarted?: EventCallback;
  onDemoStopped?: EventCallback;
  onPatientArrived?: EventCallback;
  onProcessingStarted?: EventCallback;
  onProcessingProgress?: EventCallback;
  onProcessingComplete?: EventCallback;
  onWorklistUpdated?: EventCallback;
  onError?: EventCallback;
  onPhase1Complete?: EventCallback;
  onModelSwapping?: EventCallback;
  onPhase2Started?: EventCallback;
  onPhase2Complete?: EventCallback;
  onDemoComplete?: EventCallback;
}

export function useTriageWebSocket(options: UseTriageWebSocketOptions = {}) {
  const [connected, setConnected] = useState(false);
  const [lastEvent, setLastEvent] = useState<WSEvent | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<number | null>(null);
  const optionsRef = useRef(options);

  // Keep options ref updated
  useEffect(() => {
    optionsRef.current = options;
  }, [options]);

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/triage`;

    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;

    ws.onopen = () => {
      console.log('WebSocket connected');
      setConnected(true);
    };

    ws.onclose = () => {
      console.log('WebSocket disconnected');
      setConnected(false);

      // Reconnect after delay
      reconnectTimeoutRef.current = window.setTimeout(() => {
        connect();
      }, 3000);
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
    };

    ws.onmessage = (event) => {
      try {
        const wsEvent: WSEvent = JSON.parse(event.data);
        setLastEvent(wsEvent);

        // Route to appropriate callback
        const callbacks: Record<string, EventCallback | undefined> = {
          demo_started: optionsRef.current.onDemoStarted,
          demo_stopped: optionsRef.current.onDemoStopped,
          patient_arrived: optionsRef.current.onPatientArrived,
          processing_started: optionsRef.current.onProcessingStarted,
          processing_progress: optionsRef.current.onProcessingProgress,
          processing_complete: optionsRef.current.onProcessingComplete,
          worklist_updated: optionsRef.current.onWorklistUpdated,
          error: optionsRef.current.onError,
          phase1_complete: optionsRef.current.onPhase1Complete,
          model_swapping: optionsRef.current.onModelSwapping,
          phase2_started: optionsRef.current.onPhase2Started,
          phase2_complete: optionsRef.current.onPhase2Complete,
          demo_complete: optionsRef.current.onDemoComplete,
        };

        const callback = callbacks[wsEvent.event];
        if (callback) {
          callback(wsEvent.data);
        }
      } catch (err) {
        console.error('Failed to parse WebSocket message:', err);
      }
    };
  }, []);

  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }

    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }

    setConnected(false);
  }, []);

  // Connect on mount, disconnect on unmount
  useEffect(() => {
    connect();

    return () => {
      disconnect();
    };
  }, [connect, disconnect]);

  // Ping to keep connection alive
  useEffect(() => {
    if (!connected) return;

    const pingInterval = setInterval(() => {
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        wsRef.current.send('ping');
      }
    }, 30000);

    return () => clearInterval(pingInterval);
  }, [connected]);

  return {
    connected,
    lastEvent,
    connect,
    disconnect,
  };
}
