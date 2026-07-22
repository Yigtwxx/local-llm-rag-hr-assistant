import { ArrowUpRight } from 'lucide-react';

interface SuggestionChipsProps {
  suggestions: string[];
  onSelect: (question: string) => void;
  disabled?: boolean;
}

/**
 * Follow-up questions offered under the latest answer.
 *
 * Every chip is answerable: the questions were written against passages that
 * are in the index and reviewed by hand before being committed. That is the
 * point of the whole detour through `data/suggested-questions.yaml` — a chip
 * the assistant would then refuse teaches the user not to trust the chips.
 *
 * Plain buttons in document order, so Tab reaches them without any roving
 * tabindex: there is no grid to navigate here, just a short row after the text
 * someone has finished reading.
 */
export function SuggestionChips({
  suggestions,
  onSelect,
  disabled = false,
}: SuggestionChipsProps) {
  if (suggestions.length === 0) return null;

  return (
    <div className="mt-4">
      {/* Full `text-muted-foreground`, not `/70`: the alpha took this to
          4.08:1 on the dark background, under WCAG AA for 11px text. */}
      <p className="mb-2 text-[11px] tracking-wide text-muted-foreground">
        Şunları da sorabilirsiniz
      </p>
      <ul className="flex flex-wrap gap-2">
        {suggestions.map((suggestion, index) => (
          <li key={suggestion}>
            <button
              type="button"
              onClick={() => onSelect(suggestion)}
              disabled={disabled}
              // Radius and elevation come from the active model's skin rather
              // than from hardcoded classes, so the chips change shape with
              // everything else when the model is switched.
              className="elevate animate-rise group/chip flex items-center gap-1.5 rounded-[var(--radius)] border border-border bg-card px-3 py-1.5 text-xs text-muted-foreground transition-colors duration-200 hover:border-primary/30 hover:text-foreground focus-visible:ring-2 focus-visible:ring-ring focus-visible:outline-none disabled:pointer-events-none disabled:opacity-50"
              style={{ animationDelay: `${index * 60}ms` }}
            >
              <span>{suggestion}</span>
              <ArrowUpRight
                className="size-3 shrink-0 opacity-0 transition-opacity duration-200 group-hover/chip:opacity-60"
                aria-hidden
              />
            </button>
          </li>
        ))}
      </ul>
    </div>
  );
}
