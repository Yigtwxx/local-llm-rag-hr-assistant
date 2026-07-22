import { FileText, Search, Type } from 'lucide-react';
import SpotlightCard from '@/components/SpotlightCard';
import { Badge } from '@/components/ui/badge';
import type { Source } from '@/lib/types';

interface SourceListProps {
  sources: Source[];
  retrievalMs?: number | undefined;
  /**
   * `rail` is the persistent right-hand column; `inline` is the copy that
   * expands under an answer on narrow screens, where the rail is hidden.
   */
  variant?: 'rail' | 'inline';
}

/**
 * Cosine similarity has no intuitive scale, so the bar carries the meaning
 * and the number stays for anyone who wants it. The thresholds match the
 * retrieval calibration: 0,46 is the configured cut-off, so anything near it
 * is a weak match that only just qualified.
 */
function scoreTone(score: number): { text: string; bar: string } {
  if (score >= 0.62) return { text: 'text-success', bar: 'bg-success' };
  // The middle band's number is neutral while its bar keeps the chart colour.
  // `text-chart-2` measured 3.68:1 on the light background — the chart tokens
  // are tuned to be distinguishable from each other as fills, not to be legible
  // as 12px type, which is the same trap hardcoded Tailwind colours fell into
  // here before. Strong and weak still speak through `--success`/`--warning`.
  if (score >= 0.5) return { text: 'text-foreground', bar: 'bg-chart-2' };
  return { text: 'text-warning', bar: 'bg-warning' };
}

/**
 * Chunks are embedded with their document title and section prepended, so the
 * excerpt opens by repeating the two lines directly above it. Stripping that
 * prefix hands the four-line clamp back to the passage itself.
 */
function trimEcho(excerpt: string, ...prefixes: string[]): string {
  let text = excerpt.trimStart();
  for (const prefix of prefixes) {
    const candidate = prefix.trim();
    if (candidate && text.startsWith(candidate)) {
      text = text.slice(candidate.length).trimStart();
    }
  }
  return text;
}

function ScoreMeter({ score }: { score: number }) {
  const tone = scoreTone(score);
  // Scores live in a narrow band around the 0,46 threshold; mapping 0,3–0,8
  // to the full width makes the differences between passages legible.
  const pct = Math.min(100, Math.max(6, ((score - 0.3) / 0.5) * 100));

  return (
    <div className="flex shrink-0 flex-col items-end gap-1">
      <span className={`tabular font-mono text-xs font-medium ${tone.text}`}>
        {score.toFixed(3)}
      </span>
      <div
        className="h-1 w-12 overflow-hidden rounded-full bg-muted"
        role="img"
        aria-label={`Kosinüs benzerliği ${score.toFixed(3)}`}
      >
        <div className={`h-full rounded-full ${tone.bar}`} style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}

export function SourceList({ sources, retrievalMs, variant = 'rail' }: SourceListProps) {
  if (sources.length === 0) {
    return <p className="text-sm text-muted-foreground">Henüz kaynak yok.</p>;
  }

  return (
    <div className="space-y-2.5">
      {variant === 'rail' && retrievalMs !== undefined && (
        <div className="flex justify-end">
          <span className="tabular flex items-center gap-1 font-mono text-xs text-muted-foreground">
            <Search className="size-3" aria-hidden />
            {retrievalMs.toFixed(0)} ms
          </span>
        </div>
      )}

      {sources.map((source, index) => (
        <SpotlightCard
          key={`${source.source_file}-${source.section}-${index}`}
          className="animate-rise p-3.5 transition-shadow duration-200 ease-out hover:elevate-lift"
          spotlightColor="rgba(224, 169, 42, 0.12)"
          // Stagger so the list resolves top-to-bottom instead of all at once.
          style={{ animationDelay: `${index * 60}ms` }}
        >
          <div className="flex items-start justify-between gap-3">
            <div className="flex min-w-0 gap-2.5">
              {/* The top passage carries the same gold the logo uses for the
                  selected layer, so the mark and the list say one thing. */}
              <span
                className={`tabular mt-0.5 flex size-4.5 shrink-0 items-center justify-center rounded font-mono text-[10px] font-medium ${
                  index === 0
                    ? 'bg-gold/20 text-gold'
                    : 'bg-muted text-muted-foreground'
                }`}
              >
                {index + 1}
              </span>
              <div className="min-w-0">
                {/* Section leads: every passage from one policy shares a
                    document title, so the title alone cannot tell two cards
                    apart — the section is the discriminator. */}
                <p className="truncate text-sm font-medium" title={source.section}>
                  {source.section}
                </p>
                <p
                  className="mt-0.5 truncate text-xs text-muted-foreground"
                  title={source.doc_title}
                >
                  {source.doc_title}
                </p>
              </div>
            </div>
            <ScoreMeter score={source.score} />
          </div>

          {/* `break-words` for the same reason as the answer bubbles: the rail
              is only 16rem wide, so an unbroken run in a passage escapes the
              card long before it would escape the transcript. */}
          <p className="mt-2.5 line-clamp-4 pl-7 text-xs leading-relaxed break-words text-muted-foreground">
            {trimEcho(source.excerpt, source.doc_title, source.section)}
          </p>

          <div className="mt-2.5 ml-7 flex flex-wrap items-center gap-1.5">
            <Badge
              variant="secondary"
              className="gap-1 font-mono text-[10px] font-normal"
            >
              <FileText className="size-3" aria-hidden />
              {source.source_file}
            </Badge>

            {/* Word matches are here despite a low similarity score, not
                because of one. Without this the meter reads as "weak match"
                and quietly undermines a passage that answers the question
                word for word. */}
            {source.matched_by === 'lexical' && (
              <Badge
                variant="secondary"
                className="gap-1 text-[10px] font-normal"
                title="Bu parça, sorudaki nadir bir kelimeyi birebir içerdiği için getirildi. Benzerlik skoru düşük görünür; eşleşme kelime düzeyindedir."
              >
                <Type className="size-3" aria-hidden />
                kelime eşleşmesi
              </Badge>
            )}
          </div>
        </SpotlightCard>
      ))}
    </div>
  );
}
