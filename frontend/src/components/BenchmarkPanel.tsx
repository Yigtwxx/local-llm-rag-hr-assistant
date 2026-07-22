import { useEffect, useState } from 'react';
import { ArrowDown, Trophy } from 'lucide-react';
import CountUp from '@/components/CountUp';
import { fetchBenchmark } from '@/lib/api';
import type { BenchResponse, BenchSummary } from '@/lib/types';

/** Each model keeps one colour across every metric so rows stay comparable. */
const MODEL_COLORS = ['bg-chart-1', 'bg-chart-2', 'bg-chart-3'] as const;

/** Horizontal bar comparing one metric across models. */
function MetricBar({
  label,
  value,
  max,
  unit,
  color,
  decimals = 0,
  lowerIsBetter = false,
}: {
  label: string;
  value: number | undefined;
  max: number;
  unit: string;
  color: string;
  decimals?: number;
  lowerIsBetter?: boolean;
}) {
  if (value === undefined) return null;
  const pct = max > 0 ? Math.max(4, (value / max) * 100) : 0;

  return (
    <div className="space-y-1.5">
      <div className="flex items-baseline justify-between gap-2 text-xs">
        <span className="flex items-center gap-1 text-muted-foreground">
          {label}
          {/* Bar length tracks the raw value, so for latency and memory a
              longer bar is worse. Say so rather than let the chart imply
              the opposite. */}
          {lowerIsBetter && (
            <ArrowDown className="size-3 opacity-50" aria-label="düşük olan iyi" />
          )}
        </span>
        <span className="tabular font-mono font-medium">
          <CountUp to={Number(value.toFixed(decimals))} duration={1.1} />
          <span className="ml-1 font-normal text-muted-foreground">{unit}</span>
        </span>
      </div>
      <div className="h-1.5 overflow-hidden rounded-full bg-muted">
        <div
          className={`h-full rounded-full ${color} transition-[width] duration-700 ease-out`}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
}

function SummaryCard({
  summary,
  peers,
  index,
}: {
  summary: BenchSummary;
  peers: BenchSummary[];
  index: number;
}) {
  const maxSpeed = Math.max(...peers.map((s) => s.tokens_per_second_weighted ?? 0), 1);
  const maxTtft = Math.max(...peers.map((s) => s.median_ttft_ms ?? 0), 1);
  const maxMemory = Math.max(...peers.map((s) => s.reported_memory_gb ?? 0), 1);
  const color = MODEL_COLORS[index % MODEL_COLORS.length] ?? 'bg-chart-1';

  const qualityPct =
    summary.quality_total > 0
      ? Math.round((summary.quality_passed / summary.quality_total) * 100)
      : 0;

  // How far ahead of the next model this one is on throughput.
  const speed = summary.tokens_per_second_weighted;
  const rivalBest = Math.max(
    ...peers
      .filter((s) => s.model !== summary.model)
      .map((s) => s.tokens_per_second_weighted ?? 0),
    0,
  );
  const lead =
    speed !== undefined && rivalBest > 0 && speed > rivalBest
      ? Math.round((speed / rivalBest - 1) * 100)
      : undefined;

  return (
    <div className="elevate space-y-3 rounded-xl border border-border bg-card p-3.5">
      <div className="flex items-center justify-between gap-2">
        <span className="flex min-w-0 items-center gap-2">
          <span className={`size-2 shrink-0 rounded-full ${color}`} aria-hidden />
          <span className="truncate font-mono text-xs font-medium">{summary.model}</span>
        </span>
        <span
          className={`tabular shrink-0 font-mono text-xs ${
            qualityPct >= 80
              ? 'text-success'
              : qualityPct >= 60
                ? 'text-warning'
                : 'text-destructive'
          }`}
        >
          {summary.quality_passed}/{summary.quality_total} doğru
        </span>
      </div>

      {lead !== undefined && (
        <span className="inline-flex items-center gap-1 rounded-md bg-success/10 px-2 py-0.5 text-[11px] font-medium text-success">
          <Trophy className="size-3" aria-hidden />%{lead} daha hızlı
        </span>
      )}

      <MetricBar
        label="Üretim hızı"
        value={speed}
        max={maxSpeed}
        unit="tok/s"
        color={color}
        decimals={1}
      />
      <MetricBar
        label="İlk cevap (TTFT)"
        value={summary.median_ttft_ms}
        max={maxTtft}
        unit="ms"
        color={color}
        lowerIsBetter
      />
      <MetricBar
        label="Bellek"
        value={summary.reported_memory_gb}
        max={maxMemory}
        unit="GB"
        color={color}
        decimals={1}
        lowerIsBetter
      />

      <div className="tabular space-y-1 border-t border-border pt-2.5 text-[11px] text-muted-foreground">
        <div className="flex justify-between">
          <span>Kaynağa sadakat</span>
          <span className="font-mono">
            {summary.grounding_passed}/{summary.grounding_total}
          </span>
        </div>
        {summary.mean_tokens_per_second !== undefined && (
          <div className="flex justify-between">
            <span>Vaka ortalaması</span>
            <span className="font-mono">
              {summary.mean_tokens_per_second.toFixed(1)}
              {summary.stdev_tokens_per_second !== undefined &&
                ` ± ${summary.stdev_tokens_per_second.toFixed(1)}`}{' '}
              tok/s
            </span>
          </div>
        )}
      </div>
    </div>
  );
}

export function BenchmarkPanel() {
  const [data, setData] = useState<BenchResponse | undefined>(undefined);

  useEffect(() => {
    fetchBenchmark()
      .then(setData)
      .catch(() => setData({ available: false }));
  }, []);

  const summaries = data?.summaries ?? [];

  return (
    <section className="space-y-2.5">
      {summaries.length === 0 ? (
        <p className="text-sm text-muted-foreground">Henüz ölçüm yok.</p>
      ) : (
        <>
          {summaries.map((summary, index) => (
            <SummaryCard
              key={summary.model}
              summary={summary}
              peers={summaries}
              index={index}
            />
          ))}
          {data?.generated_at && (
            <p className="pt-1 text-[11px] text-muted-foreground">
              Ölçüm: {new Date(data.generated_at).toLocaleString('tr-TR')}
            </p>
          )}
        </>
      )}
    </section>
  );
}
