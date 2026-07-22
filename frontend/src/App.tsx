import { useCallback, useEffect, useState } from 'react';
import { useReducedMotion } from 'motion/react';
import { Moon, PlugZap, Sun } from 'lucide-react';
import { BenchmarkPanel } from '@/components/BenchmarkPanel';
import { ChatPanel } from '@/components/ChatPanel';
import { Logo } from '@/components/Logo';
import { ModelSwitcher } from '@/components/ModelSwitcher';
import {
  COVERED_AT,
  ModelTransition,
  SWEEP_SECONDS,
} from '@/components/ModelTransition';
import { SourceList } from '@/components/SourceList';
import { StatusBar } from '@/components/StatusBar';
import { Button } from '@/components/ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { fetchHealth, fetchModels } from '@/lib/api';
import { SKIN_CLASSES, skinFor } from '@/lib/modelSkin';
import type { HealthResponse, ModelInfo, ModelsResponse, Source } from '@/lib/types';

/**
 * Storage is behind try/catch on both sides: Safari's private mode throws on
 * access rather than returning null, and a theme preference is not worth
 * taking the app down for.
 *
 * The same key and the same fallback order are duplicated in
 * `public/theme-init.js`, which applies the class before first paint. It sits
 * in its own file rather than inline in `index.html` so the CSP can stay on
 * `script-src 'self'`. If either side changes, both have to.
 */
const THEME_KEY = 'novatek-theme';

function readStoredTheme(): 'dark' | 'light' | undefined {
  try {
    const value = localStorage.getItem(THEME_KEY);
    return value === 'dark' || value === 'light' ? value : undefined;
  } catch {
    return undefined;
  }
}

function storeTheme(theme: 'dark' | 'light'): void {
  try {
    localStorage.setItem(THEME_KEY, theme);
  } catch {
    // Preference simply does not persist this session.
  }
}

function useTheme() {
  const [dark, setDark] = useState(() => {
    const stored = readStoredTheme();
    if (stored) return stored === 'dark';
    return window.matchMedia('(prefers-color-scheme: dark)').matches;
  });

  useEffect(() => {
    document.documentElement.classList.toggle('dark', dark);
    storeTheme(dark ? 'dark' : 'light');
  }, [dark]);

  return { dark, toggle: () => setDark((value) => !value) };
}

export default function App() {
  const { dark, toggle } = useTheme();
  const reduced = useReducedMotion();
  const [health, setHealth] = useState<HealthResponse | undefined>(undefined);
  const [models, setModels] = useState<ModelsResponse | undefined>(undefined);
  const [activeModel, setActiveModel] = useState('');
  const [switchingTo, setSwitchingTo] = useState<ModelInfo | undefined>(undefined);
  const [sources, setSources] = useState<Source[]>([]);
  const [retrievalMs, setRetrievalMs] = useState<number | undefined>(undefined);
  const [busy, setBusy] = useState(false);
  const [bootState, setBootState] = useState<'loading' | 'ready' | 'error'>('loading');

  // Both calls are treated as one: without /models there is no chat panel to
  // render, so failing either way leaves the same unusable page. Swallowing
  // these — as this once did — left the status strip checking forever and the
  // main column blank, with nothing saying the backend was down.
  const loadBackend = useCallback(() => {
    setBootState('loading');
    Promise.all([fetchHealth(), fetchModels()])
      .then(([healthResponse, modelsResponse]) => {
        setHealth(healthResponse);
        setModels(modelsResponse);
        const preferred =
          modelsResponse.models.find((model) => model.available) ??
          modelsResponse.models[0];
        if (preferred) setActiveModel(preferred.name);
        setBootState('ready');
      })
      .catch(() => setBootState('error'));
  }, []);

  useEffect(() => {
    loadBackend();
  }, [loadBackend]);

  const active = models?.models.find((model) => model.name === activeModel);

  // The skin lives on <html> so it reaches portalled content and the scrollbar
  // colours too, not just the React subtree.
  useEffect(() => {
    const root = document.documentElement;
    root.classList.remove(...SKIN_CLASSES.filter(Boolean));
    const { className } = skinFor(active);
    if (className) root.classList.add(className);
  }, [active]);

  // Swap the palette at the moment the curtain fully covers the page. The
  // curtain reports its own exit, so only this one instant needs a timer.
  useEffect(() => {
    if (!switchingTo) return;
    const commit = setTimeout(
      () => setActiveModel(switchingTo.name),
      SWEEP_SECONDS * COVERED_AT * 1000,
    );
    return () => clearTimeout(commit);
  }, [switchingTo]);

  const selectModel = useCallback(
    (name: string) => {
      if (name === activeModel || switchingTo) return;
      const next = models?.models.find((model) => model.name === name);
      if (!next) return;
      // Reduced motion gets the same outcome without the sweep.
      if (reduced) setActiveModel(name);
      else setSwitchingTo(next);
    },
    [activeModel, models, reduced, switchingTo],
  );

  const handleSources = useCallback((next: Source[], ms: number | undefined) => {
    setSources(next);
    setRetrievalMs(ms);
  }, []);

  return (
    <div className="grid h-dvh grid-rows-[auto_auto_minmax(0,1fr)] bg-background text-foreground">
      <ModelTransition
        target={switchingTo}
        onDone={() => setSwitchingTo(undefined)}
      />

      <header className="border-b border-border">
        <div className="mx-auto flex max-w-[110rem] flex-wrap items-center justify-between gap-3 px-5 py-3">
          <div className="flex items-center gap-2.5">
            <Logo
              className="size-6 text-primary transition-colors duration-300"
              variant={skinFor(active).logo}
            />
            <h1 className="text-[15px] font-semibold tracking-tight">
              NovaTek İK Asistanı
            </h1>
          </div>

          <div className="flex items-center gap-2">
            {models && (
              <ModelSwitcher
                models={models.models}
                active={activeModel}
                disabled={busy || switchingTo !== undefined}
                onSelect={selectModel}
              />
            )}
            <Button
              size="icon"
              variant="ghost"
              onClick={toggle}
              aria-label={dark ? 'Aydınlık temaya geç' : 'Koyu temaya geç'}
            >
              {dark ? <Sun className="size-4" /> : <Moon className="size-4" />}
            </Button>
          </div>
        </div>
      </header>

      {/* Readiness gets its own band. Buried in the header it read as
          decoration; on its own strip it reads as system state. */}
      <div className="border-b border-border bg-card/50">
        <div className="mx-auto max-w-[110rem] px-5 py-2">
          <StatusBar
            health={health}
            embeddingModel={models?.embedding_model}
            unreachable={bootState === 'error'}
          />
        </div>
      </div>

      {/* The rail widens with the window instead of stopping at the shell's
          centring gutter — on a wide screen that gutter left a dead strip to
          its right. The chat column absorbs the rest; its own max-w-3xl keeps
          the reading measure fixed.

          `min-h-0` on both children is load-bearing, not tidying. A grid item
          defaults to `min-height: auto`, which refuses to shrink below its
          content — so once a conversation grew past the viewport the column
          grew with it, the `overflow-y-auto` inside never received a bounded
          height to scroll within, and `overflow-hidden` here clipped the rest.
          The transcript above the fold and the composer below it both became
          unreachable, with no scrollbar to say so. */}
      <main className="mx-auto grid w-full max-w-[110rem] grid-cols-1 overflow-hidden lg:grid-cols-[minmax(0,1fr)_16rem] xl:grid-cols-[minmax(0,1fr)_20rem]">
        <div className="min-h-0 min-w-0">
          {bootState === 'error' ? (
            <div className="flex h-full flex-col items-center justify-center gap-4 px-5 text-center">
              <PlugZap className="size-8 text-destructive/70" aria-hidden />
              <div className="space-y-1.5">
                <h2 className="text-base font-semibold">Sunucuya ulaşılamadı</h2>
                <p className="max-w-sm text-sm text-muted-foreground">
                  Arka uç yanıt vermiyor. <code className="font-mono">./scripts/dev.sh</code>{' '}
                  veya <code className="font-mono">docker compose up</code> ile
                  başlattıktan sonra tekrar deneyin.
                </p>
              </div>
              <Button variant="outline" onClick={loadBackend}>
                Tekrar dene
              </Button>
            </div>
          ) : (
            activeModel &&
            models && (
              <ChatPanel
                model={activeModel}
                models={models.models}
                onSources={handleSources}
                onBusyChange={setBusy}
              />
            )
          )}
        </div>

        <aside className="hidden min-h-0 border-l border-border bg-sidebar/40 lg:block">
          <Tabs defaultValue="sources" className="h-full gap-0">
            <TabsList variant="line" className="w-full px-4 pt-3 pb-1">
              <TabsTrigger value="sources">
                Kaynaklar
                {sources.length > 0 && (
                  <span className="tabular ml-1 rounded bg-muted px-1.5 font-mono text-[10px]">
                    {sources.length}
                  </span>
                )}
              </TabsTrigger>
              <TabsTrigger value="bench">Benchmark</TabsTrigger>
            </TabsList>

            <TabsContent
              value="sources"
              className="scroll-slim overflow-y-auto px-4 py-4"
            >
              <SourceList sources={sources} retrievalMs={retrievalMs} />
            </TabsContent>

            <TabsContent value="bench" className="scroll-slim overflow-y-auto px-4 py-4">
              <BenchmarkPanel />
            </TabsContent>
          </Tabs>
        </aside>
      </main>
    </div>
  );
}
