/** Shapes mirrored from the FastAPI backend (`app/schemas.py`). */

export interface Source {
  doc_title: string;
  section: string;
  source_file: string;
  score: number;
  excerpt: string;
  /**
   * Which arm of retrieval found this passage. `lexical` means a rare word in
   * the question matched it, and its similarity score is expected to look low.
   */
  matched_by: 'dense' | 'lexical';
}

/**
 * A measurement missing from these two arrives over the wire as `null`, not as
 * an absent key — Python has no `undefined`. `lib/api.ts` converts it on the
 * way in, so by the time a component sees one of these it really is
 * `undefined`. Read them through those functions; a raw `response.json()` here
 * would put nulls back and `!== undefined` does not catch them.
 */
export interface Metrics {
  model: string;
  ttft_ms: number | undefined;
  total_ms: number | undefined;
  eval_count: number | undefined;
  tokens_per_second: number | undefined;
  retrieval_ms: number | undefined;
}

export interface ModelInfo {
  name: string;
  role: 'primary' | 'secondary';
  available: boolean;
}

export interface ModelsResponse {
  models: ModelInfo[];
  embedding_model: string;
}

export interface HealthResponse {
  ollama_reachable: boolean;
  ollama_version: string | undefined;
  collection_ready: boolean;
  indexed_chunks: number;
  available_models: string[];
  configured_models: string[];
  missing_models: string[];
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  /** Passages this specific answer was grounded in. */
  sources?: Source[];
  /** How long retrieval took for this answer, in milliseconds. */
  retrievalMs?: number;
  /** The question that produced this answer — lets it be re-asked. */
  question?: string;
  /** Model that produced this answer; may differ from the active one. */
  model?: string;
  metrics?: Metrics;
  /** False when the assistant declined because retrieval found nothing. */
  grounded?: boolean;
  /**
   * Follow-up questions drawn from passages ranked near this answer. Each one
   * is answerable: they were written against passages that are in the index.
   */
  suggestions?: string[];
  streaming?: boolean;
  /**
   * The user cancelled this answer. Distinct from `error`: nothing went wrong,
   * and whatever text had already arrived is still valid — it is just not the
   * whole answer, which the reader cannot tell from the text alone.
   */
  stopped?: boolean;
  error?: string;
}

/** One benchmarked model's aggregate results. Nulls are normalised as above. */
export interface BenchSummary {
  model: string;
  runs: number;
  tokens_per_second_weighted: number | undefined;
  mean_tokens_per_second: number | undefined;
  stdev_tokens_per_second: number | undefined;
  median_ttft_ms: number | undefined;
  reported_memory_gb: number | undefined;
  peak_memory_gb: number | undefined;
  quality_passed: number;
  quality_total: number;
  grounding_passed: number;
  grounding_total: number;
  load_seconds: number | undefined;
}

export interface BenchResponse {
  available: boolean;
  hint?: string;
  generated_at?: string;
  hardware?: Record<string, unknown>;
  settings?: Record<string, unknown>;
  summaries?: BenchSummary[];
}
