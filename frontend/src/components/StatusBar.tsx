import { Brain, Database, ShieldCheck } from 'lucide-react';
import type { HealthResponse } from '@/lib/types';

interface StatusBarProps {
  health: HealthResponse | undefined;
  embeddingModel: string | undefined;
  /** The backend could not be reached at all — distinct from "still checking". */
  unreachable?: boolean;
}

function Detail({
  icon: Icon,
  children,
  mono = false,
}: {
  icon: typeof Brain;
  children: React.ReactNode;
  mono?: boolean;
}) {
  return (
    <span
      className={`flex items-center gap-1.5 text-muted-foreground ${mono ? 'font-mono' : ''}`}
    >
      <Icon className="size-3.5 opacity-70" aria-hidden />
      {children}
    </span>
  );
}

/**
 * Compact readiness strip.
 *
 * Surfacing this is the point of a local-first tool: if Ollama is down or the
 * index is empty, the user should see why rather than get a silent failure.
 */
export function StatusBar({ health, embeddingModel, unreachable }: StatusBarProps) {
  // A failed probe used to be indistinguishable from a slow one, so the strip
  // sat on "checking…" indefinitely while the page stayed empty.
  if (unreachable) {
    return (
      <div className="flex items-center gap-2 text-xs font-medium text-destructive">
        <span className="size-1.5 rounded-full bg-destructive" />
        Sunucuya ulaşılamadı
      </div>
    );
  }

  if (!health) {
    return (
      <div className="flex items-center gap-2 text-xs text-muted-foreground">
        <span className="size-1.5 animate-pulse rounded-full bg-muted-foreground" />
        Durum kontrol ediliyor…
      </div>
    );
  }

  const problems: string[] = [];
  if (!health.ollama_reachable) problems.push('Ollama çalışmıyor');
  if (!health.collection_ready) problems.push('İndeks boş');
  if (health.missing_models.length > 0) {
    problems.push(`Eksik model: ${health.missing_models.join(', ')}`);
  }

  const healthy = problems.length === 0;

  return (
    <div className="tabular flex flex-wrap items-center gap-x-4 gap-y-1.5 text-xs">
      {healthy ? (
        <span className="flex items-center gap-1.5 font-medium text-success">
          <span className="relative flex size-1.5">
            <span className="absolute inline-flex size-full animate-ping rounded-full bg-success opacity-60" />
            <span className="relative inline-flex size-1.5 rounded-full bg-success" />
          </span>
          <ShieldCheck className="size-3.5" aria-hidden />
          Tamamen lokal · hazır
        </span>
      ) : (
        <span className="flex items-center gap-1.5 font-medium text-warning">
          <span className="size-1.5 rounded-full bg-warning" />
          {problems.join(' · ')}
        </span>
      )}

      <span className="hidden h-3 w-px bg-border sm:block" />

      <Detail icon={Database}>{health.indexed_chunks} parça indeksli</Detail>

      {embeddingModel && (
        <Detail icon={Brain} mono>
          {embeddingModel}
        </Detail>
      )}
    </div>
  );
}
