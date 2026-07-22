import { Component } from 'react';
import type { ErrorInfo, ReactNode } from 'react';
import { TriangleAlert } from 'lucide-react';
import { Button } from '@/components/ui/button';

interface ErrorBoundaryProps {
  children: ReactNode;
}

interface ErrorBoundaryState {
  failed: boolean;
}

/**
 * Last line of defence around the app.
 *
 * React unmounts the entire tree when a render throws, so without a boundary
 * any single bad value anywhere produces a blank white page with nothing on it
 * to explain or recover from — which is exactly what one unreadable metric did.
 * This does not excuse the bug that got here; it bounds the damage to a screen
 * the user can read and act on.
 *
 * Still a class component: `getDerivedStateFromError` and `componentDidCatch`
 * have no hook equivalent.
 */
export class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  state: ErrorBoundaryState = { failed: false };

  static getDerivedStateFromError(): ErrorBoundaryState {
    return { failed: true };
  }

  componentDidCatch(error: Error, info: ErrorInfo): void {
    // The stack is the only trace of what happened — nothing leaves the machine,
    // so the console is where it has to land.
    console.error('Beklenmeyen hata:', error, info.componentStack);
  }

  render(): ReactNode {
    if (!this.state.failed) return this.props.children;

    return (
      <div className="flex h-dvh flex-col items-center justify-center gap-4 bg-background px-5 text-center text-foreground">
        <TriangleAlert className="size-8 text-destructive/70" aria-hidden />
        <div className="space-y-1.5">
          <h2 className="text-base font-semibold">Bir şeyler ters gitti</h2>
          <p className="max-w-sm text-sm text-muted-foreground">
            Arayüz beklenmedik bir hatayla karşılaştı. Ayrıntı tarayıcı
            konsolunda.
          </p>
        </div>
        <Button variant="outline" onClick={() => window.location.reload()}>
          Yeniden yükle
        </Button>
      </div>
    );
  }
}
