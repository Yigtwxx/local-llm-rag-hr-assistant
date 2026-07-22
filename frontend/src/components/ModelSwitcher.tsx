import { useRef } from 'react';
import { motion, useReducedMotion } from 'motion/react';
import type { ModelInfo } from '@/lib/types';

const STEP_BY_KEY: Record<string, number> = {
  ArrowRight: 1,
  ArrowDown: 1,
  ArrowLeft: -1,
  ArrowUp: -1,
};

interface ModelSwitcherProps {
  models: ModelInfo[];
  active: string;
  disabled: boolean;
  onSelect: (model: string) => void;
}

/**
 * Switches between the two benchmarked models at runtime.
 *
 * Being able to re-ask the same question on the other model is what makes the
 * benchmark tangible rather than a table in a report.
 */
export function ModelSwitcher({ models, active, disabled, onSelect }: ModelSwitcherProps) {
  const reduced = useReducedMotion();
  const buttons = useRef<Array<HTMLButtonElement | null>>([]);

  /**
   * A radio group is expected to move with the arrow keys, not with Tab —
   * Tab enters the group and leaves it. Unavailable models are skipped, and
   * the search wraps, so the two ends of the group are neighbours.
   */
  const handleKeyDown = (event: React.KeyboardEvent, from: number) => {
    const step = STEP_BY_KEY[event.key];
    if (step === undefined || disabled) return;
    event.preventDefault();

    const count = models.length;
    for (let hop = 1; hop <= count; hop++) {
      const index = (((from + step * hop) % count) + count) % count;
      const candidate = models[index];
      if (!candidate?.available) continue;
      onSelect(candidate.name);
      buttons.current[index]?.focus();
      return;
    }
  };

  return (
    <div
      role="radiogroup"
      aria-label="Model seçimi"
      className="inline-flex rounded-lg border border-border bg-muted/50 p-0.5"
    >
      {models.map((model, index) => {
        const selected = model.name === active;
        return (
          <button
            key={model.name}
            ref={(element) => {
              buttons.current[index] = element;
            }}
            role="radio"
            aria-checked={selected}
            // Roving tabindex: the group is one stop in the tab order, and the
            // arrow keys move within it.
            tabIndex={selected ? 0 : -1}
            disabled={disabled || !model.available}
            onClick={() => onSelect(model.name)}
            onKeyDown={(event) => handleKeyDown(event, index)}
            title={model.available ? model.name : `${model.name} indirilmemiş`}
            className={`relative rounded-[max(1px,calc(var(--radius)-3px))] px-3 py-1.5 font-mono text-xs transition-colors duration-200 focus-visible:ring-2 focus-visible:ring-ring focus-visible:outline-none disabled:cursor-not-allowed disabled:opacity-40 ${
              selected ? 'text-foreground' : 'text-muted-foreground hover:text-foreground'
            }`}
          >
            {selected && (
              <motion.span
                layoutId="model-pill"
                className="absolute inset-0 rounded-[max(1px,calc(var(--radius)-3px))] elevate border border-border bg-background"
                transition={
                  reduced ? { duration: 0 } : { type: 'spring', stiffness: 420, damping: 34 }
                }
              />
            )}
            <span className="relative">{model.name}</span>
          </button>
        );
      })}
    </div>
  );
}
