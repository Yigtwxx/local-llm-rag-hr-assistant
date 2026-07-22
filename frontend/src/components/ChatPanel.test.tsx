import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { ChatPanel } from '@/components/ChatPanel';
import { streamChat } from '@/lib/api';
import type { ChatHandlers } from '@/lib/api';
import type { Metrics, ModelInfo, Source } from '@/lib/types';

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
  matched_by: 'dense',
};

const METRICS = {
  model: 'qwen3.5:9b',
  ttft_ms: 120,
  total_ms: 900,
  eval_count: 42,
  tokens_per_second: 47.5,
  retrieval_ms: 30,
};

// What the backend actually sends when it refuses: retrieval found nothing, so
// no generation call was made and every stat is absent. `app/api.py` renders
// those absent values as JSON `null`, not as a missing key — which is exactly
// what the earlier version of this file got wrong by reusing METRICS here.
const REFUSAL_METRICS = JSON.parse(
  JSON.stringify({
    model: 'qwen3.5:9b',
    ttft_ms: null,
    total_ms: null,
    eval_count: null,
    tokens_per_second: null,
    retrieval_ms: 21,
  }),
) as Metrics;

/** Drives the last stream the panel opened, as the backend would. */
function lastHandlers(): ChatHandlers {
  const call = streamChatMock.mock.calls.at(-1);
  if (!call) throw new Error('streamChat was never called');
  return call[2];
}

function renderPanel() {
  const onSources = vi.fn();
  const { container } = render(
    <ChatPanel
      model="qwen3.5:9b"
      models={MODELS}
      onSources={onSources}
      onBusyChange={() => undefined}
    />,
  );
  return { onSources, container };
}

/**
 * Drive the transcript's scroll container the way the browser would.
 *
 * jsdom has no layout, so the three geometry properties are defined here and
 * a `scroll` event is dispatched exactly as a real scroll does.
 */
function scrollTo(
  container: HTMLElement,
  geometry: { scrollTop: number; scrollHeight: number; clientHeight: number },
) {
  const el = container.querySelector('.scroll-slim');
  if (!el) throw new Error('scroll container not found');
  for (const [key, value] of Object.entries(geometry)) {
    Object.defineProperty(el, key, { value, configurable: true });
  }
  fireEvent.scroll(el);
}

/** Asks the first starter question and streams a complete answer back. */
async function askAndAnswer(
  user: ReturnType<typeof userEvent.setup>,
  suggestions: string[] = [],
) {
  await user.click(screen.getByRole('button', { name: /yıllık izin hakkı/i }));

  const handlers = lastHandlers();
  handlers.onSources([SOURCE], 43);
  handlers.onToken('Yıllık izniniz 16 iş günüdür.');
  handlers.onDone(METRICS, true, suggestions);
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

  it('asks a follow-up when its chip is clicked', async () => {
    const user = userEvent.setup();
    renderPanel();
    await askAndAnswer(user, ['Evlilik izni kaç gün?']);

    await user.click(await screen.findByRole('button', { name: /evlilik izni/i }));

    const [question, model] = streamChatMock.mock.calls.at(-1) ?? [];
    expect(question).toBe('Evlilik izni kaç gün?');
    expect(model).toBe('qwen3.5:9b');
  });

  it('drops the previous answer chips once a new question is asked', async () => {
    const user = userEvent.setup();
    renderPanel();
    await askAndAnswer(user, ['Evlilik izni kaç gün?']);

    await user.click(await screen.findByRole('button', { name: /evlilik izni/i }));

    // The chips belong to the newest answer. Leaving the old row on screen
    // while a new answer streams offers routes out of a question the user has
    // already moved past.
    await waitFor(() => {
      expect(
        screen.queryByRole('button', { name: /evlilik izni/i }),
      ).not.toBeInTheDocument();
    });
  });

  it('offers follow-ups even when the assistant refuses', async () => {
    const user = userEvent.setup();
    renderPanel();

    await user.click(screen.getByRole('button', { name: /yıllık izin hakkı/i }));
    const handlers = lastHandlers();
    handlers.onSources([], 21);
    handlers.onToken('Bu bilgi elimdeki İK dokümanlarında yer almıyor.');
    // A refusal is where a user has least idea what the assistant does know,
    // so the chips matter more here than after a successful answer.
    handlers.onDone(REFUSAL_METRICS, false, ['Harcırah ne kadar?']);

    expect(
      await screen.findByRole('button', { name: /harcırah ne kadar/i }),
    ).toBeInTheDocument();
  });

  it('renders a refusal whose metrics are all absent', async () => {
    const user = userEvent.setup();
    renderPanel();

    await user.click(screen.getByRole('button', { name: /yıllık izin hakkı/i }));
    const handlers = lastHandlers();
    handlers.onSources([], 21);
    handlers.onToken('Bu bilgi elimdeki İK dokümanlarında yer almıyor.');
    handlers.onDone(REFUSAL_METRICS, false, []);

    // The refusal itself has to survive. Reading a stat that is not there used
    // to throw during render, and with no boundary above it that took the whole
    // app down — a blank page for every out-of-scope question.
    expect(
      await screen.findByText(/İK dokümanlarında yer almıyor/),
    ).toBeInTheDocument();
    expect(screen.getByText('qwen3.5:9b')).toBeInTheDocument();
    // Nothing was generated, so there is no speed or TTFT to show.
    expect(screen.queryByText(/tok\/s/)).not.toBeInTheDocument();
  });

  it('starts one stream when two sends land in the same tick', async () => {
    renderPanel();

    // Two clicks inside one task both saw `busy === false`, because state is
    // only visible to the next render. Both streams started, the second
    // overwrote `abortRef`, and the first became uncancellable — still pushing
    // its tokens into whatever bubble happened to be last.
    const starter = screen.getByRole('button', { name: /yıllık izin hakkı/i });
    starter.click();
    starter.click();
    starter.click();

    await waitFor(() => expect(streamChatMock).toHaveBeenCalled());
    expect(streamChatMock).toHaveBeenCalledTimes(1);
  });

  it('accepts the next question once the previous answer finishes', async () => {
    const user = userEvent.setup();
    renderPanel();

    // The other half of the guard: whatever locks the composer has to unlock
    // it again, or the panel accepts exactly one question per page load.
    await askAndAnswer(user);
    await user.click(await screen.findByRole('button', { name: /gemma4:12b ile sor/i }));

    expect(streamChatMock).toHaveBeenCalledTimes(2);
  });

  it('accepts the next question after a cancelled answer', async () => {
    const user = userEvent.setup();
    renderPanel();

    await user.click(screen.getByRole('button', { name: /yıllık izin hakkı/i }));
    await user.keyboard('{Escape}');
    await user.click(await screen.findByRole('button', { name: 'Tekrar dene' }));

    expect(streamChatMock).toHaveBeenCalledTimes(2);
  });

  it('accepts the next question after a failed answer', async () => {
    const user = userEvent.setup();
    renderPanel();

    await user.click(screen.getByRole('button', { name: /yıllık izin hakkı/i }));
    lastHandlers().onError('Sunucuya ulaşılamadı.');
    await user.click(await screen.findByRole('button', { name: 'Tekrar dene' }));

    expect(streamChatMock).toHaveBeenCalledTimes(2);
  });

  it('lets an unbroken run of text wrap instead of escaping its bubble', async () => {
    const user = userEvent.setup();
    renderPanel();

    const link = `https://intranet.novatek.example/ik/${'a'.repeat(250)}`;
    await user.click(screen.getByRole('button', { name: /yıllık izin hakkı/i }));
    lastHandlers().onToken(link);
    lastHandlers().onDone(METRICS, true, []);

    // jsdom does not lay out, so the invariant is checked where it is declared.
    // Measured in Chrome before this: a 3798px-wide paragraph inside a 728px
    // bubble, running across the page and under the sources rail.
    const paragraph = await screen.findByText(link);
    expect(paragraph).toHaveClass('break-words');

    const question = screen.getByText(/3 yıldır çalışan/);
    expect(question).toHaveClass('break-words');
  });

  it('leaves follow-the-stream on when the container shrinks under it', async () => {
    const user = userEvent.setup();
    const { container } = renderPanel();

    await user.click(screen.getByRole('button', { name: /yıllık izin hakkı/i }));
    lastHandlers().onToken('Yıllık izniniz');

    // The empty state is four starter cards tall; the transcript that replaces
    // it holds one short question. The container stops overflowing, the browser
    // clamps scrollTop to fit and reports it as a scroll. Read as a gesture,
    // that switched follow-the-stream off for the rest of every first answer.
    scrollTo(container, { scrollTop: 78, scrollHeight: 404, clientHeight: 326 });
    scrollTo(container, { scrollTop: 0, scrollHeight: 326, clientHeight: 326 });

    expect(
      screen.queryByRole('button', { name: 'En alta in' }),
    ).not.toBeInTheDocument();
  });

  it('lets any upward gesture stop the view following the stream', async () => {
    const user = userEvent.setup();
    const { container } = renderPanel();

    await user.click(screen.getByRole('button', { name: /yıllık izin hakkı/i }));
    lastHandlers().onToken('Yıllık izniniz 16 iş günüdür.');

    // A finger drag, a dragged scrollbar and Page Up all scroll without ever
    // firing `wheel`, which is the only gesture this used to listen for — on a
    // phone the transcript could not be read while an answer streamed.
    scrollTo(container, { scrollTop: 600, scrollHeight: 1400, clientHeight: 700 });
    scrollTo(container, { scrollTop: 0, scrollHeight: 1400, clientHeight: 700 });

    expect(await screen.findByRole('button', { name: 'En alta in' })).toBeInTheDocument();

    // Returning to the bottom re-engages it.
    scrollTo(container, { scrollTop: 700, scrollHeight: 1400, clientHeight: 700 });
    await waitFor(() => {
      expect(
        screen.queryByRole('button', { name: 'En alta in' }),
      ).not.toBeInTheDocument();
    });
  });

  it('says an answer was stopped and hands the question back', async () => {
    const user = userEvent.setup();
    renderPanel();

    await user.click(screen.getByRole('button', { name: /yıllık izin hakkı/i }));
    // Cancelled before the first token: no text, no stats, no sources, no
    // error. The bubble used to render as a blank card with nothing on it —
    // the question was gone and had to be typed again from scratch.
    await user.keyboard('{Escape}');

    expect(await screen.findByText('Yanıt durduruldu.')).toBeInTheDocument();

    await user.click(await screen.findByRole('button', { name: 'Tekrar dene' }));

    const [question, model] = streamChatMock.mock.calls.at(-1) ?? [];
    expect(question).toBe('3 yıldır çalışan birinin yıllık izin hakkı kaç gün?');
    expect(model).toBe('qwen3.5:9b');
  });

  it('marks a partial answer as stopped rather than letting it read as complete', async () => {
    const user = userEvent.setup();
    renderPanel();

    await user.click(screen.getByRole('button', { name: /yıllık izin hakkı/i }));
    lastHandlers().onToken('Yıllık izniniz');
    await user.keyboard('{Escape}');

    expect(await screen.findByText('Yanıt yarıda durduruldu.')).toBeInTheDocument();
    // The text that did arrive is still there; only its completeness is in doubt.
    expect(screen.getByText('Yıllık izniniz')).toBeInTheDocument();
  });

  it('reports a clipboard failure instead of failing silently', async () => {
    const user = userEvent.setup();
    // Chrome rejects the write whenever the document is not focused. That
    // rejection was unhandled, so the button did nothing and said nothing.
    Object.defineProperty(navigator, 'clipboard', {
      value: { writeText: vi.fn().mockRejectedValue(new Error('not focused')) },
      configurable: true,
    });

    renderPanel();
    await askAndAnswer(user);

    await user.click(await screen.findByRole('button', { name: 'Kopyala' }));

    expect(await screen.findByRole('button', { name: 'Kopyalanamadı' })).toBeInTheDocument();
  });

  it('survives a context where the clipboard API does not exist', async () => {
    const user = userEvent.setup();
    // Any `http://` origin that is not localhost — a LAN address, for one.
    // Reading `.writeText` off `undefined` threw inside the click handler.
    Object.defineProperty(navigator, 'clipboard', {
      value: undefined,
      configurable: true,
    });

    renderPanel();
    await askAndAnswer(user);

    await user.click(await screen.findByRole('button', { name: 'Kopyala' }));

    expect(await screen.findByRole('button', { name: 'Kopyalanamadı' })).toBeInTheDocument();
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
