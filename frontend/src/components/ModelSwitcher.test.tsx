import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';
import { ModelSwitcher } from '@/components/ModelSwitcher';
import type { ModelInfo } from '@/lib/types';

const MODELS: ModelInfo[] = [
  { name: 'qwen3.5:9b', role: 'primary', available: true },
  { name: 'gemma4:12b', role: 'secondary', available: true },
];

function renderSwitcher(models: ModelInfo[] = MODELS) {
  const onSelect = vi.fn();
  render(
    <ModelSwitcher
      models={models}
      active="qwen3.5:9b"
      disabled={false}
      onSelect={onSelect}
    />,
  );
  return { onSelect };
}

describe('ModelSwitcher', () => {
  it('keeps one tab stop for the whole group', () => {
    renderSwitcher();

    expect(screen.getByRole('radio', { name: 'qwen3.5:9b' })).toHaveAttribute(
      'tabindex',
      '0',
    );
    expect(screen.getByRole('radio', { name: 'gemma4:12b' })).toHaveAttribute(
      'tabindex',
      '-1',
    );
  });

  it('moves to the next model with an arrow key', async () => {
    const user = userEvent.setup();
    const { onSelect } = renderSwitcher();

    screen.getByRole('radio', { name: 'qwen3.5:9b' }).focus();
    await user.keyboard('{ArrowRight}');

    expect(onSelect).toHaveBeenCalledWith('gemma4:12b');
  });

  it('wraps around at the end of the group', async () => {
    const user = userEvent.setup();
    const { onSelect } = renderSwitcher();

    screen.getByRole('radio', { name: 'qwen3.5:9b' }).focus();
    await user.keyboard('{ArrowLeft}');

    expect(onSelect).toHaveBeenCalledWith('gemma4:12b');
  });

  it('skips a model that is not downloaded', async () => {
    const user = userEvent.setup();
    const { onSelect } = renderSwitcher([
      MODELS[0] as ModelInfo,
      { name: 'missing:1b', role: 'secondary', available: false },
    ]);

    screen.getByRole('radio', { name: 'qwen3.5:9b' }).focus();
    await user.keyboard('{ArrowRight}');

    // The only other member is unavailable, so the search wraps back to the
    // model already selected rather than landing on a disabled control.
    expect(onSelect).toHaveBeenCalledWith('qwen3.5:9b');
  });
});
