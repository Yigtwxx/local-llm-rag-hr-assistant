import { useCallback, useEffect, useRef, useState } from 'react';
import { ArrowDown, ArrowUp, ArrowUpRight, Square } from 'lucide-react';
import BorderGlow from '@/components/BorderGlow';
import { Logo } from '@/components/Logo';
import { MessageBubble } from '@/components/MessageBubble';
import ShinyText from '@/components/ShinyText';
import SpotlightCard from '@/components/SpotlightCard';
import { SuggestionChips } from '@/components/SuggestionChips';
import { Button } from '@/components/ui/button';
import { streamChat } from '@/lib/api';
import { skinFor } from '@/lib/modelSkin';
import type { ChatMessage, ModelInfo, Source } from '@/lib/types';

// Shown on the empty screen, before there is anything to follow up on. The
// chips that appear *after* an answer are a different thing entirely: they come
// from the backend, chosen from the passages ranked near that answer.
const STARTERS = [
  '3 yıldır çalışan birinin yıllık izin hakkı kaç gün?',
  'Haftada kaç gün ofisten çalışmam gerekiyor?',
  'Yurt içi seyahatte günlük harcırah ne kadar?',
  'Yıllık eğitim bütçem ne kadar, devreder mi?',
];

let messageCounter = 0;
const nextId = () => `m${++messageCounter}`;

interface ChatPanelProps {
  model: string;
  models: ModelInfo[];
  onSources: (sources: Source[], retrievalMs: number | undefined) => void;
  onBusyChange: (busy: boolean) => void;
}

export function ChatPanel({ model, models, onSources, onBusyChange }: ChatPanelProps) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [draft, setDraft] = useState('');
  const [busy, setBusy] = useState(false);
  const [focusedId, setFocusedId] = useState<string | undefined>(undefined);
  const [atBottom, setAtBottom] = useState(true);
  const abortRef = useRef<(() => void) | undefined>(undefined);
  const scrollRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    onBusyChange(busy);
  }, [busy, onBusyChange]);

  // Raise whichever answer is focused into the right-hand rail. Deriving this
  // from state rather than firing it at call sites keeps a single source of
  // truth: the rail always shows the passages of the focused message.
  useEffect(() => {
    const focused = messages.find((message) => message.id === focusedId);
    onSources(focused?.sources ?? [], focused?.retrievalMs);
  }, [focusedId, messages, onSources]);

  useEffect(() => {
    if (!atBottom) return;
    scrollRef.current?.scrollTo({
      top: scrollRef.current.scrollHeight,
      behavior: 'smooth',
    });
  }, [messages, atBottom]);

  // Cancel any in-flight stream when the panel unmounts.
  useEffect(() => () => abortRef.current?.(), []);

  // The composer is the only thing anyone comes here to use.
  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  // Hand focus back when an answer finishes, so a follow-up needs no click.
  // Only on the falling edge: focusing on every render would pull the caret
  // away from whatever the user was reading while the answer streamed.
  const wasBusy = useRef(false);
  useEffect(() => {
    if (!busy && wasBusy.current) inputRef.current?.focus();
    wasBusy.current = busy;
  }, [busy]);

  // Follow-the-stream is disengaged by an upward gesture and re-engaged near
  // the bottom. Direction is read from the scroll position rather than from the
  // wheel, which was the only gesture this used to listen for: a touch drag, a
  // dragged scrollbar and Page Up all scroll without ever firing `wheel`, so on
  // a phone the transcript could not be read while an answer streamed — every
  // token pulled the view straight back down.
  //
  // Deriving direction is safe even though the programmatic follow scroll fires
  // `scroll` as well: that one only ever moves *towards* the bottom, so a
  // decreasing scrollTop is always the user's doing.
  const lastScrollTop = useRef(0);
  const lastMaxScroll = useRef(0);
  const handleScroll = () => {
    const el = scrollRef.current;
    if (!el) return;
    const { scrollTop, scrollHeight, clientHeight } = el;
    const maxScroll = Math.max(0, scrollHeight - clientHeight);

    // A container that shrinks clamps `scrollTop` down to fit and reports that
    // as a `scroll` event, indistinguishable from a gesture by position alone.
    // It happens on the first question of every session: the empty state, four
    // starter cards tall, is replaced by a transcript holding one short
    // question. On a phone-sized viewport that collapse moved scrollTop 78 → 0
    // and follow-the-stream switched itself off for the rest of the answer.
    // The tell is that the scrollable range shrank too — the layout moved, not
    // the reader.
    const clamped = maxScroll < lastMaxScroll.current;

    if (!clamped && scrollTop < lastScrollTop.current - 1) setAtBottom(false);
    else if (maxScroll - scrollTop < 80) setAtBottom(true);

    lastScrollTop.current = scrollTop;
    lastMaxScroll.current = maxScroll;
  };

  const patchLast = useCallback((patch: Partial<ChatMessage>) => {
    setMessages((prev) => {
      if (prev.length === 0) return prev;
      const next = [...prev];
      const last = next[next.length - 1];
      if (!last) return prev;
      next[next.length - 1] = { ...last, ...patch };
      return next;
    });
  }, []);

  /**
   * Guarded on a ref, not on the `busy` state it mirrors.
   *
   * `busy` is only visible to the next render, so two sends dispatched inside
   * one task both read `false` and both start a stream. The second overwrites
   * `abortRef`, which strands the first: nothing can cancel it any more and its
   * tokens keep arriving into `patchLast`, appending one answer's text onto a
   * different question's bubble. A ref closes the window at the point the
   * decision is made rather than one render later.
   */
  const sending = useRef(false);

  const finish = useCallback(() => {
    sending.current = false;
    setBusy(false);
  }, []);

  const send = (question: string, modelOverride?: string) => {
    const trimmed = question.trim();
    if (!trimmed || sending.current) return;
    sending.current = true;

    const target = modelOverride ?? model;
    const answerId = nextId();

    setDraft('');
    setBusy(true);
    setAtBottom(true);
    setFocusedId(answerId);
    setMessages((prev) => [
      ...prev,
      { id: nextId(), role: 'user', content: trimmed },
      {
        id: answerId,
        role: 'assistant',
        content: '',
        streaming: true,
        question: trimmed,
        model: target,
      },
    ]);

    if (inputRef.current) inputRef.current.style.height = 'auto';

    abortRef.current = streamChat(trimmed, target, {
      onSources: (sources, retrievalMs) => patchLast({ sources, retrievalMs }),
      onToken: (text) =>
        setMessages((prev) => {
          const next = [...prev];
          const last = next[next.length - 1];
          if (!last) return prev;
          next[next.length - 1] = { ...last, content: last.content + text };
          return next;
        }),
      onDone: (metrics, grounded, suggestions) => {
        patchLast({ streaming: false, metrics, grounded, suggestions });
        finish();
      },
      onError: (message) => {
        patchLast({ streaming: false, error: message });
        finish();
      },
    });
  };

  // `stopped` rather than just clearing `streaming`: cancelled before the first
  // token, the answer has no text, no stats, no sources and no error, so the
  // bubble rendered as a blank card with nothing on it to read or act on — the
  // question was gone and the only way back was to type it again. The flag is
  // what lets the bubble say it was stopped and offer the question back.
  const stop = useCallback(() => {
    abortRef.current?.();
    patchLast({ streaming: false, stopped: true });
    finish();
  }, [finish, patchLast]);

  // Esc cancels a running answer — the stop button is a mouse trip away, and
  // a wrong question is usually obvious within the first line. `stop` is
  // stable, so the listener is bound once per answer rather than per token.
  useEffect(() => {
    if (!busy) return;
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') stop();
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [busy, stop]);

  const otherModelFor = (used: string | undefined) =>
    models.find((candidate) => candidate.available && candidate.name !== (used ?? model))
      ?.name;

  const skin = skinFor(models.find((candidate) => candidate.name === model));

  // Chips belong to the newest answer, and only once it has finished streaming.
  const newest = messages[messages.length - 1];
  const lastAnswer =
    newest?.role === 'assistant' && !newest.streaming && !newest.error
      ? newest
      : undefined;

  return (
    <div className="relative flex h-full flex-col">
      <div
        ref={scrollRef}
        onScroll={handleScroll}
        className="scroll-slim flex-1 overflow-y-auto"
      >
        <div className="mx-auto flex min-h-full w-full max-w-3xl flex-col px-5 py-6">
          {messages.length === 0 ? (
            <div className="flex flex-1 flex-col items-center justify-center gap-7 text-center">
              <Logo className="size-9 text-primary/70" variant={skin.logo} />

              <h2 className="text-2xl font-semibold tracking-tight">
                <ShinyText
                  text="Ne öğrenmek istersiniz?"
                  speed={2.6}
                  delay={1.6}
                  spread={110}
                  color="var(--shiny-base)"
                  shineColor="var(--shiny-shine)"
                />
              </h2>

              {/* The two skins offer the same four questions in the shape
                  that suits them: floating cards under the soft skin, a
                  numbered index under the flat one. */}
              {skin.suggestions === 'cards' ? (
                <div className="grid w-full max-w-xl gap-2.5 sm:grid-cols-2">
                  {STARTERS.map((suggestion, index) => (
                    <SpotlightCard
                      key={suggestion}
                      className="animate-rise transition-all duration-200 ease-out hover:elevate-lift hover:border-primary/30 motion-safe:hover:-translate-y-0.5 motion-safe:active:translate-y-0"
                      spotlightColor={skin.spotlight}
                      style={{ animationDelay: `${index * 70}ms` }}
                    >
                      <button
                        onClick={() => send(suggestion)}
                        className="h-full w-full px-3.5 py-3 text-left text-xs leading-relaxed text-muted-foreground transition-colors duration-200 hover:text-foreground focus-visible:ring-2 focus-visible:ring-ring focus-visible:outline-none"
                      >
                        {suggestion}
                      </button>
                    </SpotlightCard>
                  ))}
                </div>
              ) : (
                <ul className="w-full max-w-xl divide-y divide-border border-y border-border text-left">
                  {STARTERS.map((suggestion, index) => (
                    <li
                      key={suggestion}
                      className="animate-rise"
                      style={{ animationDelay: `${index * 70}ms` }}
                    >
                      <button
                        onClick={() => send(suggestion)}
                        className="group/row flex w-full items-center gap-3 px-3 py-3 text-left text-xs leading-relaxed text-muted-foreground transition-colors duration-200 hover:bg-accent hover:text-foreground focus-visible:ring-2 focus-visible:ring-ring focus-visible:outline-none"
                      >
                        {/* Same reason as the chip label: at 10px an alpha on
                            top of the muted token lands well under WCAG AA.
                            `aria-hidden` keeps it out of the a11y tree but it
                            is still text a person has to read. */}
                        <span
                          className="tabular shrink-0 font-mono text-[10px] text-muted-foreground"
                          aria-hidden
                        >
                          {String(index + 1).padStart(2, '0')}
                        </span>
                        <span className="flex-1">{suggestion}</span>
                        <ArrowUpRight
                          className="size-3.5 shrink-0 opacity-0 transition-opacity duration-200 group-hover/row:opacity-60"
                          aria-hidden
                        />
                      </button>
                    </li>
                  ))}
                </ul>
              )}
            </div>
          ) : (
            <div className="space-y-6">
              {messages.map((message) => (
                <MessageBubble
                  key={message.id}
                  message={message}
                  otherModel={otherModelFor(message.model)}
                  onReask={send}
                  onFocusSources={setFocusedId}
                  focused={message.id === focusedId}
                />
              ))}

              {/* Only under the newest answer. Chips on every past message
                  would turn the transcript into a wall of buttons and offer
                  routes out of a question the user has already moved past. */}
              <SuggestionChips
                suggestions={lastAnswer?.suggestions ?? []}
                onSelect={send}
                disabled={busy}
              />
            </div>
          )}
        </div>
      </div>

      {/* Anchored to the composer, not to the panel. A fixed offset from the
          bottom assumed a fixed composer height and it has neither: on a phone
          it landed on the metrics row and covered the tokens-per-second
          reading, and once the textarea grew towards its 160px cap the button
          ended up behind the composer entirely, still visible and no longer
          clickable. `bottom-full` keeps the same gap above whatever height the
          composer currently has. */}
      <div className="relative shrink-0 px-5 pb-4">
        {!atBottom && messages.length > 0 && (
          <button
            type="button"
            onClick={() => setAtBottom(true)}
            aria-label="En alta in"
            className="avatar-shape elevate-lift absolute bottom-full left-1/2 z-10 mb-3 flex size-8 -translate-x-1/2 items-center justify-center border border-border bg-card text-muted-foreground transition-colors hover:text-foreground focus-visible:ring-2 focus-visible:ring-ring focus-visible:outline-none"
          >
            <ArrowDown className="size-4" aria-hidden />
          </button>
        )}

        <div className="mx-auto w-full max-w-3xl">
          <BorderGlow
            borderRadius={skin.composerRadius}
            glowRadius={24}
            coneSpread={30}
            glowIntensity={0.55}
            fillOpacity={0.35}
            colors={skin.glow}
            animated={busy}
            className="w-full"
          >
            <div className="flex items-end gap-2 p-2">
              <textarea
                ref={inputRef}
                // Named so the browser can associate the field with itself
                // across reloads; without either attribute Chrome reports the
                // composer as an unidentified form field.
                id="question"
                name="question"
                rows={1}
                value={draft}
                onChange={(event) => {
                  setDraft(event.target.value);
                  const el = event.target;
                  el.style.height = 'auto';
                  el.style.height = `${Math.min(el.scrollHeight, 160)}px`;
                }}
                onKeyDown={(event) => {
                  if (event.key === 'Enter' && !event.shiftKey) {
                    event.preventDefault();
                    send(draft);
                  }
                }}
                placeholder="Bir soru sorun…"
                className="max-h-40 flex-1 resize-none bg-transparent px-2.5 py-2 text-sm outline-none placeholder:text-muted-foreground"
              />
              {busy ? (
                <Button size="icon" variant="secondary" onClick={stop} aria-label="Durdur">
                  <Square className="size-3.5" aria-hidden />
                </Button>
              ) : (
                <Button
                  size="icon"
                  onClick={() => send(draft)}
                  disabled={!draft.trim()}
                  aria-label="Gönder"
                >
                  <ArrowUp className="size-4" aria-hidden />
                </Button>
              )}
            </div>
          </BorderGlow>
        </div>
      </div>
    </div>
  );
}
