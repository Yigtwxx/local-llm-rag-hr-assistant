import type { LogoVariant } from '@/lib/modelSkin';

/**
 * Brand mark: three stacked passages, the front one selected.
 *
 * It draws what the product actually does — pick the right chunk out of many —
 * rather than borrowing the generic spark/orb vocabulary. The two rear layers
 * are hollow and fading back; only the front one is filled, and the gold rule
 * inside it is the same gold the app uses everywhere else to mean "this is the
 * passage the answer came from".
 *
 * Both layer colours ride on `currentColor`, so the mark inverts with the
 * theme without a second asset.
 *
 * The two variants are the same idea under the two form languages, not two
 * logos: `staggered` offsets the layers and rounds them, matching the soft
 * skin; `flush` squares them off and stacks them left-aligned, matching the
 * flat one. Depth is carried by width instead of offset there, so the stack
 * still reads as a stack.
 */
export function Logo({
  className,
  variant = 'staggered',
}: {
  className?: string;
  variant?: LogoVariant;
}) {
  const flush = variant === 'flush';

  // Gaps are kept to ~1/4 of a layer's height. Any airier and the three
  // bars stop reading as one stack.
  const rx = flush ? 0.4 : 1.7;
  const back = flush ? { x: 3, width: 13.5 } : { x: 8.75, width: 12.25 };
  const middle = flush ? { x: 3, width: 15.75 } : { x: 5.875, width: 12.25 };
  const front = flush ? { x: 3, width: 18 } : { x: 3, width: 12.25 };

  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      className={className}
      role="img"
      aria-label="NovaTek İK Asistanı"
    >
      <rect
        x={back.x}
        y="3"
        width={back.width}
        height="5"
        rx={rx}
        stroke="currentColor"
        strokeWidth="1.5"
        opacity="0.28"
      />
      <rect
        x={middle.x}
        y="9.2"
        width={middle.width}
        height="5"
        rx={rx}
        stroke="currentColor"
        strokeWidth="1.5"
        opacity="0.55"
      />
      <rect
        x={front.x}
        y="15.4"
        width={front.width}
        height="5.2"
        rx={rx}
        fill="currentColor"
      />
      <rect
        x="5.3"
        y="17.4"
        width="4.6"
        height="1.2"
        rx={flush ? 0 : 0.6}
        fill="var(--color-gold)"
      />
    </svg>
  );
}
