import { useState } from 'react';
import {
  AlertTriangle,
  Check,
  ChevronDown,
  Clock,
  Copy,
  FileStack,
  Gauge,
  RefreshCw,
  ShieldCheck,
  User,
} from 'lucide-react';
import ShinyText from '@/components/ShinyText';
import { SourceList } from '@/components/SourceList';
import type { ChatMessage } from '@/lib/types';

/**
 * Minimal Markdown rendering.
 *
 * The models reply with bold text, bullets and the occasional table. Pulling in
 * a full Markdown pipeline for that is disproportionate, so we handle the three
 * constructs that actually appear and render everything else as plain text —
 * which also means no untrusted HTML is ever injected.
 */
function renderInline(text: string): React.ReactNode[] {
  return text.split(/(\*\*[^*]+\*\*|`[^`]+`)/g).map((part, index) => {
    if (part.startsWith('**') && part.endsWith('**')) {
      return (
        <strong key={index} className="font-semibold text-foreground">
          {part.slice(2, -2)}
        </strong>
      );
    }
    if (part.startsWith('`') && part.endsWith('`')) {
      return (
        <code
          key={index}
          className="rounded border border-border bg-muted px-1 py-0.5 font-mono text-[0.85em]"
        >
          {part.slice(1, -1)}
        </code>
      );
    }
    return part;
  });
}

function AnswerBody({ content }: { content: string }) {
  const lines = content.split('\n');
  const blocks: React.ReactNode[] = [];
  let bullets: string[] = [];

  const flushBullets = () => {
    if (bullets.length === 0) return;
    blocks.push(
      <ul key={`ul-${blocks.length}`} className="my-2.5 ml-0.5 space-y-2">
        {bullets.map((item, index) => (
          <li key={index} className="flex gap-2.5">
            <span className="avatar-shape mt-[0.55em] size-1.5 shrink-0 bg-primary/50" />
            <span>{renderInline(item)}</span>
          </li>
        ))}
      </ul>,
    );
    bullets = [];
  };

  for (const line of lines) {
    const trimmed = line.trim();
    const bullet = /^[-*•]\s+(.*)$/.exec(trimmed) ?? /^\d+[.)]\s+(.*)$/.exec(trimmed);

    if (bullet?.[1] !== undefined) {
      bullets.push(bullet[1]);
      continue;
    }
    flushBullets();

    if (trimmed === '') continue;

    // Table rows are shown as-is in a monospace strip; converting them to a
    // real <table> mid-stream would reflow on every token.
    if (trimmed.startsWith('|')) {
      blocks.push(
        <p
          key={blocks.length}
          className="tabular my-0.5 overflow-x-auto font-mono text-xs whitespace-pre text-muted-foreground"
        >
          {trimmed}
        </p>,
      );
      continue;
    }

    blocks.push(
      <p key={blocks.length} className="my-2 leading-[1.7]">
        {renderInline(trimmed)}
      </p>,
    );
  }
  flushBullets();

  return <>{blocks}</>;
}

/** One measurement, rendered so the digits do not shift as they update. */
function MetricPill({
  icon: Icon,
  value,
  label,
}: {
  icon: typeof Gauge;
  value: string;
  label: string;
}) {
  return (
    <span
      title={label}
      className="tabular inline-flex items-center gap-1.5 rounded-md bg-muted/60 px-2 py-1 font-mono text-[11px] text-muted-foreground"
    >
      <Icon className="size-3 opacity-70" aria-hidden />
      {value}
    </span>
  );
}

function ActionButton({
  icon: Icon,
  label,
  onClick,
}: {
  icon: typeof Copy;
  label: string;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      title={label}
      aria-label={label}
      className="inline-flex items-center gap-1.5 rounded-md px-2 py-1 text-[11px] text-muted-foreground transition-colors hover:bg-muted hover:text-foreground focus-visible:ring-2 focus-visible:ring-ring focus-visible:outline-none"
    >
      <Icon className="size-3" aria-hidden />
      {label}
    </button>
  );
}

interface MessageBubbleProps {
  message: ChatMessage;
  /** The model the answer was NOT produced with, if one is available. */
  otherModel?: string | undefined;
  onReask?: (question: string, model: string) => void;
  /** Raises this answer's passages into the right-hand rail. */
  onFocusSources?: (id: string) => void;
  focused?: boolean;
}

export function MessageBubble({
  message,
  otherModel,
  onReask,
  onFocusSources,
  focused = false,
}: MessageBubbleProps) {
  const [copied, setCopied] = useState(false);
  const [expanded, setExpanded] = useState(false);

  if (message.role === 'user') {
    return (
      <div className="animate-rise flex justify-end gap-3">
        <div className="elevate max-w-[80%] rounded-2xl rounded-br-md bg-primary px-4 py-2.5 text-sm leading-relaxed text-primary-foreground">
          {message.content}
        </div>
        <div className="avatar-shape mt-0.5 flex size-7 shrink-0 items-center justify-center border border-border bg-card">
          <User className="size-3.5 text-muted-foreground" aria-hidden />
        </div>
      </div>
    );
  }

  const ungrounded = message.grounded === false;
  const sources = message.sources ?? [];
  const topScore = sources.length > 0 ? Math.max(...sources.map((s) => s.score)) : undefined;

  const copy = () => {
    void navigator.clipboard.writeText(message.content).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 1600);
    });
  };

  const showActions = !message.streaming && !message.error && message.content !== '';

  return (
    <div className="animate-rise group/message flex gap-3">
      <div
        className={`avatar-shape mt-0.5 flex size-7 shrink-0 items-center justify-center border ${
          ungrounded
            ? 'border-warning/30 bg-warning/10'
            : 'border-primary/20 bg-primary/10'
        } ${message.streaming ? 'animate-live' : ''}`}
      >
        {ungrounded ? (
          <AlertTriangle className="size-3.5 text-warning" aria-hidden />
        ) : (
          <ShieldCheck className="size-3.5 text-primary" aria-hidden />
        )}
      </div>

      <div className="min-w-0 flex-1">
        {message.error ? (
          <div className="rounded-2xl rounded-tl-md border border-destructive/40 bg-destructive/5 px-4 py-3 text-sm text-destructive">
            {message.error}
          </div>
        ) : (
          <div
            className={`elevate rounded-2xl rounded-tl-md border px-4 py-3 text-sm ${
              ungrounded ? 'border-warning/30 bg-warning/5' : 'border-border bg-card'
            }`}
          >
            <AnswerBody content={message.content} />
            {message.streaming && (
              <span className="animate-caret ml-0.5 inline-block h-[1.1em] w-[2px] translate-y-[0.2em] bg-primary" />
            )}
          </div>
        )}

        {message.streaming && (
          <p className="mt-2 px-1 text-[11px]">
            <ShinyText
              text="yanıtlanıyor…"
              speed={1.6}
              spread={110}
              color="var(--shiny-base)"
              shineColor="var(--shiny-shine)"
            />
          </p>
        )}

        {(message.metrics || showActions) && (
          <div className="mt-2 flex flex-wrap items-center gap-1.5">
            {message.metrics && !message.streaming && (
              <>
                <span className="rounded-md bg-muted/60 px-2 py-1 font-mono text-[11px] text-muted-foreground">
                  {message.metrics.model}
                </span>
                {message.metrics.tokens_per_second !== undefined && (
                  <MetricPill
                    icon={Gauge}
                    value={`${message.metrics.tokens_per_second.toFixed(1)} tok/s`}
                    label="Üretim hızı"
                  />
                )}
                {message.metrics.ttft_ms !== undefined && (
                  <MetricPill
                    icon={Clock}
                    value={`${message.metrics.ttft_ms.toFixed(0)} ms`}
                    label="İlk token süresi (TTFT)"
                  />
                )}
              </>
            )}

            {showActions && (
              <div className="flex items-center gap-0.5 opacity-0 transition-opacity group-hover/message:opacity-100 focus-within:opacity-100">
                <ActionButton
                  icon={copied ? Check : Copy}
                  label={copied ? 'Kopyalandı' : 'Kopyala'}
                  onClick={copy}
                />
                {otherModel && message.question && onReask && (
                  <ActionButton
                    icon={RefreshCw}
                    label={`${otherModel} ile sor`}
                    onClick={() => onReask(message.question ?? '', otherModel)}
                  />
                )}
              </div>
            )}
          </div>
        )}

        {sources.length > 0 && !message.streaming && (
          <div className="mt-2">
            <button
              type="button"
              onClick={() => {
                setExpanded((value) => !value);
                onFocusSources?.(message.id);
              }}
              aria-expanded={expanded}
              className={`inline-flex items-center gap-1.5 rounded-md border px-2 py-1 text-[11px] transition-colors ${
                focused
                  ? 'border-primary/40 bg-primary/10 text-primary'
                  : 'border-border text-muted-foreground hover:border-primary/30 hover:text-foreground'
              }`}
            >
              <FileStack className="size-3" aria-hidden />
              {sources.length} kaynak
              {topScore !== undefined && (
                <span className="tabular font-mono opacity-70">
                  · en yüksek {topScore.toFixed(2)}
                </span>
              )}
              <ChevronDown
                className={`size-3 transition-transform lg:hidden ${
                  expanded ? 'rotate-180' : ''
                }`}
                aria-hidden
              />
            </button>

            {/* Below `lg` the rail is hidden, so the passages expand in place. */}
            {expanded && (
              <div className="mt-2.5 lg:hidden">
                <SourceList
                  sources={sources}
                  retrievalMs={message.retrievalMs}
                  variant="inline"
                />
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
