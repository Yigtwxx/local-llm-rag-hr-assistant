import type { ModelInfo } from './types';

/**
 * Per-model visual identity.
 *
 * Switching models is the one interaction that makes the benchmark tangible,
 * so it re-skins the app rather than just swapping a label. Everything here is
 * keyed off `ModelInfo.role`, which the backend supplies — parsing the model
 * name would break the moment a model is renamed or re-versioned.
 *
 * Colour is only half of it: the two skins also differ in form — corner
 * radius, elevation and edge weight — which is what makes them readable
 * without a label. Those live in `index.css` under `.skin-secondary`; this
 * module holds the handful of decisions CSS custom properties cannot express.
 */
export interface ModelSkin {
  /** Applied to <html> and to the transition overlay to re-tint the surface. */
  className: string;
  /** Mesh colours for the composer's BorderGlow. */
  glow: string[];
  /** Cursor-spotlight tint for the suggestion cards. */
  spotlight: `rgba(${number}, ${number}, ${number}, ${number})`;
  /**
   * Composer corner, in px. BorderGlow paints its own frame in inline styles
   * and cannot read `--radius`, so the one value the CSS skin owns has to be
   * mirrored here — keep the two in step.
   */
  composerRadius: number;
  /** Brand-mark construction; see `Logo`. */
  logo: LogoVariant;
  /** How the empty screen offers its starting questions. */
  suggestions: 'cards' | 'index';
}

/** Layer arrangement of the brand mark. */
export type LogoVariant = 'staggered' | 'flush';

/** The base palette in `index.css` already is the primary model's skin. */
const PRIMARY: ModelSkin = {
  className: '',
  glow: ['#2f5d8f', '#5a92cc', '#e0a92a'],
  spotlight: 'rgba(47, 93, 143, 0.13)',
  composerRadius: 16,
  logo: 'staggered',
  suggestions: 'cards',
};

const SECONDARY: ModelSkin = {
  className: 'skin-secondary',
  glow: ['#1f7d84', '#45aeb4', '#e0a92a'],
  spotlight: 'rgba(31, 125, 132, 0.13)',
  composerRadius: 3,
  logo: 'flush',
  suggestions: 'index',
};

export const SKIN_CLASSES = [SECONDARY.className];

export function skinFor(model: ModelInfo | undefined): ModelSkin {
  return model?.role === 'secondary' ? SECONDARY : PRIMARY;
}
