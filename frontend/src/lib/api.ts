import type {
  BenchResponse,
  HealthResponse,
  Metrics,
  ModelsResponse,
  Source,
} from './types';

/** Events emitted by the backend's SSE chat stream. */
type ChatEvent =
  | { type: 'sources'; sources: Source[]; retrieval_ms: number }
  | { type: 'token'; text: string }
  | { type: 'done'; grounded: boolean; metrics: Metrics }
  | { type: 'error'; message: string };

export interface ChatHandlers {
  onSources: (sources: Source[], retrievalMs: number) => void;
  onToken: (text: string) => void;
  onDone: (metrics: Metrics, grounded: boolean) => void;
  onError: (message: string) => void;
}

async function getJson<T>(path: string): Promise<T> {
  const response = await fetch(path);
  if (!response.ok) {
    throw new Error(`${path} failed with ${response.status}`);
  }
  return (await response.json()) as T;
}

export const fetchHealth = (): Promise<HealthResponse> =>
  getJson<HealthResponse>('/api/health');

export const fetchModels = (): Promise<ModelsResponse> =>
  getJson<ModelsResponse>('/api/models');

export const fetchBenchmark = (): Promise<BenchResponse> =>
  getJson<BenchResponse>('/api/bench');

/**
 * Stream an answer from the backend.
 *
 * Uses `fetch` + a manual SSE parse rather than `EventSource`, because the
 * question has to travel in a POST body and `EventSource` is GET-only.
 * Returns an abort function so a pending answer can be cancelled.
 */
export function streamChat(
  question: string,
  model: string,
  handlers: ChatHandlers,
  options: { think?: boolean } = {},
): () => void {
  const controller = new AbortController();

  void (async () => {
    try {
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question, model, think: options.think ?? false }),
        signal: controller.signal,
      });

      if (!response.ok || !response.body) {
        handlers.onError(`Sunucu ${response.status} döndü.`);
        return;
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });

        // SSE frames are separated by a blank line; the tail may be partial.
        const frames = buffer.split('\n\n');
        buffer = frames.pop() ?? '';

        for (const frame of frames) {
          const line = frame.split('\n').find((l) => l.startsWith('data: '));
          if (!line) continue;

          let event: ChatEvent;
          try {
            event = JSON.parse(line.slice(6)) as ChatEvent;
          } catch {
            continue;
          }

          switch (event.type) {
            case 'sources':
              handlers.onSources(event.sources, event.retrieval_ms);
              break;
            case 'token':
              handlers.onToken(event.text);
              break;
            case 'done':
              handlers.onDone(event.metrics, event.grounded);
              break;
            case 'error':
              handlers.onError(event.message);
              break;
          }
        }
      }
    } catch (error) {
      if (error instanceof DOMException && error.name === 'AbortError') return;
      handlers.onError(
        error instanceof Error ? error.message : 'Bağlantı kurulamadı.',
      );
    }
  })();

  return () => controller.abort();
}
