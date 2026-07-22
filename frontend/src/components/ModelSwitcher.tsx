import { motion, useReducedMotion } from 'motion/react';
import type { ModelInfo } from '@/lib/types';

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

  return (
    <div
      role="radiogroup"
      aria-label="Model seçimi"
      className="inline-flex rounded-lg border border-border bg-muted/50 p-0.5"
    >
      {models.map((model) => {
        const selected = model.name === active;
        return (
          <button
            key={model.name}
            role="radio"
            aria-checked={selected}
            disabled={disabled || !model.available}
            onClick={() => onSelect(model.name)}
            title={model.available ? model.name : `${model.name} indirilmemiş`}
            className={`relative rounded-[max(1px,calc(var(--radius)-3px))] px-3 py-1.5 font-mono text-xs transition-colors duration-200 disabled:cursor-not-allowed disabled:opacity-40 ${
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
