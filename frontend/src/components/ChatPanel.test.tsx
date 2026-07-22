import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { ChatPanel } from '@/components/ChatPanel';
import { streamChat } from '@/lib/api';
import type { ChatHandlers } from '@/lib/api';
import type { ModelInfo, Source } from '@/lib/types';

vi.mock('@/lib/api', () => ({ streamChat: vi.fn(() => () => undefined) }));

const streamChatMock = vi.mocked(streamChat);

const MODELS: ModelInfo[] = [
  { name: 'qwen3.5:9b', role: 'primary', available: true },
  { name: 'gemma4:12b', role: 'secondary', available: true },
];

const SOURCE: Source = {
  doc_title: 'İzin Politikası',
  section: '1.1 Hak Ediş',
  source_file: '01-izin.md',
  score: 0.675,
  excerpt: '16 iş günü',
};

const METRICS = {
  model: 'qwen3.5:9b',
  ttft_ms: 120,
  total_ms: 900,
  eval_count: 42,
  tokens_per_second: 47.5,
  retrieval_ms: 30,
};

/** Drives the last stream the panel opened, as the backend would. */
function lastHandlers(): ChatHandlers {
  const call = streamChatMock.mock.calls.at(-1);
  if (!call) throw new Error('streamChat was never called');
  return call[2];
}

function renderPanel() {
  const onSources = vi.fn();
  render(
    <ChatPanel
      model="qwen3.5:9b"
      models={MODELS}
      onSources={onSources}
      onBusyChange={() => undefined}
    />,
  );
  return { onSources };
}

/** Asks the first suggestion and streams a complete answer back. */
async function askAndAnswer(user: ReturnType<typeof userEvent.setup>) {
  await user.click(screen.getByRole('button', { name: /yıllık izin hakkı/i }));

  const handlers = lastHandlers();
  handlers.onSources([SOURCE], 43);
  handlers.onToken('Yıllık izniniz 16 iş günüdür.');
  handlers.onDone(METRICS, true);
}

describe('ChatPanel', () => {
  beforeEach(() => {
    streamChatMock.mockClear();
    streamChatMock.mockImplementation(() => () => undefined);
  });

  it('attaches retrieved passages to the answer that used them', async () => {
    const user = userEvent.setup();
    const { onSources } = renderPanel();

    await askAndAnswer(user);

    // The chip belongs to the message, not to a global slot — this is what
    // makes an earlier answer's grounding recoverable after a second question.
    expect(await screen.findByRole('button', { name: /1 kaynak/i })).toBeInTheDocument();

    await waitFor(() => {
      expect(onSources).toHaveBeenLastCalledWith([SOURCE], 43);
    });
  });

  it('copies the answer text to the clipboard', async () => {
    const user = userEvent.setup();
    // Defined after `setup()` — user-event installs its own clipboard stub,
    // and `navigator.clipboard` is getter-only so it cannot be assigned.
    const writeText = vi.fn().mockResolvedValue(undefined);
    Object.defineProperty(navigator, 'clipboard', {
      value: { writeText },
      configurable: true,
    });

    renderPanel();
    await askAndAnswer(user);

    await user.click(await screen.findByRole('button', { name: 'Kopyala' }));

    expect(writeText).toHaveBeenCalledWith('Yıllık izniniz 16 iş günüdür.');
  });

  it('re-asks the same question on the other model', async () => {
    const user = userEvent.setup();
    renderPanel();
    await askAndAnswer(user);

    await user.click(await screen.findByRole('button', { name: /gemma4:12b ile sor/i }));

    const [question, model] = streamChatMock.mock.calls.at(-1) ?? [];
    expect(question).toBe('3 yıldır çalışan birinin yıllık izin hakkı kaç gün?');
    expect(model).toBe('gemma4:12b');
  });

  it('cancels a streaming answer on Escape', async () => {
    const user = userEvent.setup();
    const abort = vi.fn();
    streamChatMock.mockImplementation(() => abort);

    renderPanel();
    await user.click(screen.getByRole('button', { name: /yıllık izin hakkı/i }));
    lastHandlers().onToken('Yıllık izniniz');

    await user.keyboard('{Escape}');

    expect(abort).toHaveBeenCalled();
  });

  it('retries a failed answer with the same question and model', async () => {
    const user = userEvent.setup();
    renderPanel();

    await user.click(screen.getByRole('button', { name: /yıllık izin hakkı/i }));
    lastHandlers().onError('Bağlantı kurulamadı.');

    await user.click(await screen.findByRole('button', { name: 'Tekrar dene' }));

    const [question, model] = streamChatMock.mock.calls.at(-1) ?? [];
    expect(question).toBe('3 yıldır çalışan birinin yıllık izin hakkı kaç gün?');
    expect(model).toBe('qwen3.5:9b');
  });
});
