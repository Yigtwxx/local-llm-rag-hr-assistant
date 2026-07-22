import { AnimatePresence, motion } from 'motion/react';
import { Logo } from '@/components/Logo';
import { skinFor } from '@/lib/modelSkin';
import type { ModelInfo } from '@/lib/types';

/**
 * Fraction of the sweep at which the curtain fully covers the page. The parent
 * commits the palette here; exported so the two cannot drift apart.
 */
export const SWEEP_SECONDS = 0.88;
export const COVERED_AT = 0.42;

interface ModelTransitionProps {
  /** The model being switched to, or undefined when nothing is in flight. */
  target: ModelInfo | undefined;
  /** Fired when the curtain has left the screen. */
  onDone: () => void;
}

/**
 * Curtain that covers the app while the accent palette is swapped underneath.
 *
 * The swap itself is instant — re-tinting a page mid-view would read as a
 * glitch. Sweeping a panel across hides the change and gives the switch the
 * weight it deserves: it is the moment the whole surface changes hands.
 *
 * The panel carries the *incoming* model's skin class, so it is already the
 * new colour before the app underneath is.
 */
export function ModelTransition({ target, onDone }: ModelTransitionProps) {
  return (
    <AnimatePresence>
      {target && (
        <motion.div
          key={target.name}
          // Sweeps in from the left, rests while the palette is committed, then
          // continues off to the right rather than retreating the way it came.
          initial={{ x: '-101%' }}
          animate={{ x: ['-101%', '0%', '0%', '101%'] }}
          transition={{
            duration: SWEEP_SECONDS,
            times: [0, COVERED_AT, COVERED_AT + 0.13, 1],
            ease: [0.65, 0, 0.35, 1],
          }}
          // Unmounting on the animation's own completion, rather than a timer
          // in the parent, keeps the duration defined in exactly one place.
          onAnimationComplete={onDone}
          className={`${skinFor(target).className} pointer-events-none fixed inset-0 z-50 flex items-center justify-center bg-primary`}
        >
          <motion.div
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: [0, 1, 1, 0], y: [8, 0, 0, -8] }}
            transition={{ duration: SWEEP_SECONDS, times: [0, 0.4, 0.6, 0.85] }}
            className="flex items-center gap-3 text-primary-foreground"
          >
            <Logo className="size-7" variant={skinFor(target).logo} />
            <span className="font-mono text-lg tracking-tight">{target.name}</span>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
