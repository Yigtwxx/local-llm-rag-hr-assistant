import { describe, expect, it, vi } from 'vitest';
import { streamChat } from './api';
import type { Metrics, Source } from './types';

/** Build a Response whose body streams the given chunks verbatim. */
function streamingResponse(chunks: string[]): Response {
  const encoder = new TextEncoder();
  const body = new ReadableStream<Uint8Array>({
    start(controller) {
      for (const chunk of chunks) controller.enqueue(encoder.encode(chunk));
      controller.close();
    },
  });
  return new Response(body, { status: 200 });
}

/** Records everything the stream emits and resolves once it finishes. */
function collect() {
  const tokens: string[] = [];
  const sources: Source[][] = [];
  const retrievals: number[] = [];
  const errors: string[] = [];
  let metrics: Metrics | undefined;
  let grounded: boolean | undefined;
  let finish: () => void = () => undefined;

  const done = new Promise<void>((resolve) => {
    finish = resolve;
  });

  const handlers = {
    onSources: (next: Source[], retrievalMs: number) => {
      sources.push(next);
      retrievals.push(retrievalMs);
    },
    onToken: (text: string) => tokens.push(text),
    onDone: (m: Metrics, g: boolean) => {
      metrics = m;
      grounded = g;
      finish();
    },
    onError: (message: string) => {
      errors.push(message);
      finish();
    },
  };

  return {
    handlers,
    done,
    tokens,
    sources,
    retrievals,
    errors,
    get metrics() {
      return metrics;
    },
    get grounded() {
      return grounded;
    },
  };
}

const METRICS = {
  model: 'qwen3.5:9b',
  ttft_ms: 120,
  total_ms: 900,
  eval_count: 42,
  tokens_per_second: 47.5,
  retrieval_ms: 30,
};

describe('streamChat', () => {
  it('parses sources, tokens and completion from an SSE stream', async () => {
    const source = {
      doc_title: 'İzin Politikası',
      section: '1.1 Hak Ediş',
      source_file: '01-izin.md',
      score: 0.675,
      excerpt: '16 iş günü',
    };

    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        streamingResponse([
          `data: ${JSON.stringify({ type: 'sources', sources: [source], retrieval_ms: 30 })}\n\n`,
          `data: ${JSON.stringify({ type: 'token', text: 'Yıllık ' })}\n\n`,
          `data: ${JSON.stringify({ type: 'token', text: 'izniniz 16 iş günüdür.' })}\n\n`,
          `data: ${JSON.stringify({ type: 'done', grounded: true, metrics: METRICS })}\n\n`,
        ]),
      ),
    );

    const sink = collect();
    streamChat('izin hakkım?', 'qwen3.5:9b', sink.handlers);
    await sink.done;

    expect(sink.sources[0]).toEqual([source]);
    // The retrieval timing rides along with the sources event; dropping it is
    // what previously left the "arama NN ms" readout permanently blank.
    expect(sink.retrievals[0]).toBe(30);
    expect(sink.tokens.join('')).toBe('Yıllık izniniz 16 iş günüdür.');
    expect(sink.metrics?.tokens_per_second).toBe(47.5);
    expect(sink.grounded).toBe(true);
  });

  it('reassembles frames split across network chunk boundaries', async () => {
    // A single SSE frame arriving in three pieces must not be dropped — this is
    // the failure mode that makes streamed answers lose characters.
    const frame = `data: ${JSON.stringify({ type: 'token', text: 'bölünmüş' })}\n\n`;
    const doneFrame = `data: ${JSON.stringify({ type: 'done', grounded: true, metrics: METRICS })}\n\n`;

    vi.stubGlobal(
      'fetch',
      vi
        .fn()
        .mockResolvedValue(
          streamingResponse([
            frame.slice(0, 10),
            frame.slice(10, 25),
            frame.slice(25),
            doneFrame,
          ]),
        ),
    );

    const sink = collect();
    streamChat('soru', 'qwen3.5:9b', sink.handlers);
    await sink.done;

    expect(sink.tokens.join('')).toBe('bölünmüş');
  });

  it('surfaces a backend error event', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        streamingResponse([
          `data: ${JSON.stringify({ type: 'error', message: 'Ollama kapalı' })}\n\n`,
        ]),
      ),
    );

    const sink = collect();
    streamChat('soru', 'qwen3.5:9b', sink.handlers);
    await sink.done;

    expect(sink.errors).toEqual(['Ollama kapalı']);
  });

  it('reports a non-OK HTTP status instead of hanging', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(new Response('nope', { status: 500 })),
    );

    const sink = collect();
    streamChat('soru', 'qwen3.5:9b', sink.handlers);
    await sink.done;

    expect(sink.errors[0]).toContain('500');
  });

  it('sends think=false by default so timings stay comparable', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      streamingResponse([
        `data: ${JSON.stringify({ type: 'done', grounded: true, metrics: METRICS })}\n\n`,
      ]),
    );
    vi.stubGlobal('fetch', fetchMock);

    const sink = collect();
    streamChat('soru', 'gemma4:12b', sink.handlers);
    await sink.done;

    const init = fetchMock.mock.calls[0]?.[1] as RequestInit;
    const body = JSON.parse(init.body as string) as Record<string, unknown>;
    expect(body).toMatchObject({ question: 'soru', model: 'gemma4:12b', think: false });
  });
});
