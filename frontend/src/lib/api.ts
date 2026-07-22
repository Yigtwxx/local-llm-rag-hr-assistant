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
  | { type: 'done'; grounded: boolean; metrics: Metrics; suggestions?: string[] }
  | { type: 'error'; message: string };

export interface ChatHandlers {
  onSources: (sources: Source[], retrievalMs: number) => void;
  onToken: (text: string) => void;
  onDone: (metrics: Metrics, grounded: boolean, suggestions: string[]) => void;
  onError: (message: string) => void;
}

/**
 * A measurement the backend may not have.
 *
 * Python renders an absent number as JSON `null`, but every consumer in this
 * app tests for it with `!== undefined` — and `null !== undefined` is true, so
 * the guard passes and the `.toFixed()` behind it throws mid-render. Converting
 * here, at the single point each payload is parsed, is what keeps those guards
 * honest; the alternative is remembering to write `!= null` at every call site
 * forever, and one forgotten site takes the whole page down.
 */
function numberOrUndefined(value: unknown): number | undefined {
  return typeof value === 'number' && Number.isFinite(value) ? value : undefined;
}

/** Normalise the stats block of a `done` event. `model` is not a measurement. */
function normaliseMetrics(metrics: Metrics): Metrics {
  return {
    model: metrics.model,
    ttft_ms: numberOrUndefined(metrics.ttft_ms),
    total_ms: numberOrUndefined(metrics.total_ms),
    eval_count: numberOrUndefined(metrics.eval_count),
    tokens_per_second: numberOrUndefined(metrics.tokens_per_second),
    retrieval_ms: numberOrUndefined(metrics.retrieval_ms),
  };
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

/**
 * The benchmark harness writes `null` for anything a run could not measure, and
 * `/api/bench` serves that file through verbatim — so the summaries arrive with
 * exactly the same null-versus-undefined hazard the chat metrics have.
 */
export const fetchBenchmark = async (): Promise<BenchResponse> => {
  const response = await getJson<BenchResponse>('/api/bench');
  if (!response.summaries) return response;
  return {
    ...response,
    summaries: response.summaries.map((summary) => ({
      ...summary,
      tokens_per_second_weighted: numberOrUndefined(summary.tokens_per_second_weighted),
      mean_tokens_per_second: numberOrUndefined(summary.mean_tokens_per_second),
      stdev_tokens_per_second: numberOrUndefined(summary.stdev_tokens_per_second),
      median_ttft_ms: numberOrUndefined(summary.median_ttft_ms),
      reported_memory_gb: numberOrUndefined(summary.reported_memory_gb),
      peak_memory_gb: numberOrUndefined(summary.peak_memory_gb),
      load_seconds: numberOrUndefined(summary.load_seconds),
    })),
  };
};

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
      // The stream is only over when the backend says so. A body that just
      // stops looks the same as one still arriving, so without this the caller
      // is left with a bubble that streams forever and a composer that never
      // unlocks. The backend guarantees a terminal event; this is the guard for
      // everything between here and it that could truncate the response.
      let finished = false;

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
              finished = true;
              handlers.onDone(
                normaliseMetrics(event.metrics),
                event.grounded,
                event.suggestions ?? [],
              );
              break;
            case 'error':
              finished = true;
              handlers.onError(event.message);
              break;
          }
        }
      }

      if (!finished) {
        handlers.onError('Yanıt tamamlanamadı. Lütfen tekrar deneyin.');
      }
    } catch (error) {
      if (error instanceof DOMException && error.name === 'AbortError') return;
      // Deliberately not `error.message`. A dropped connection surfaces as the
      // browser's own untranslated "Failed to fetch", which this used to put
      // straight into a Turkish interface as the entire explanation — and it
      // names the transport, not anything the reader can act on. The one thing
      // worth saying is which end went quiet.
      handlers.onError('Sunucuya ulaşılamadı. Bağlantınızı kontrol edip tekrar deneyin.');
    }
  })();

  return () => controller.abort();
}
