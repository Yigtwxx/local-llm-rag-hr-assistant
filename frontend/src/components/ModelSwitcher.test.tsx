import { render, screen, waitFor } from '@testing-library/react';
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

  it('takes focus back when the transition releases the group', async () => {
    const user = userEvent.setup();
    // Selecting starts the transition curtain, which disables the group. A
    // disabled button cannot hold focus, so the browser moved it to <body> and
    // the keyboard user was dropped out of the page with nowhere to resume.
    const { rerender } = render(
      <ModelSwitcher
        models={MODELS}
        active="qwen3.5:9b"
        disabled={false}
        onSelect={vi.fn()}
      />,
    );

    screen.getByRole('radio', { name: 'qwen3.5:9b' }).focus();
    await user.keyboard('{ArrowRight}');

    rerender(
      <ModelSwitcher
        models={MODELS}
        active="qwen3.5:9b"
        disabled
        onSelect={vi.fn()}
      />,
    );
    // Chrome blurs a focused element the moment it becomes disabled; jsdom does
    // not, so the condition under test has to be produced explicitly. Verified
    // in a real browser: focus landed on <body> after an arrow-key switch.
    (document.activeElement as HTMLElement).blur();
    expect(document.activeElement).toBe(document.body);

    rerender(
      <ModelSwitcher
        models={MODELS}
        active="gemma4:12b"
        disabled={false}
        onSelect={vi.fn()}
      />,
    );

    await waitFor(() => {
      expect(document.activeElement).toBe(screen.getByRole('radio', { name: 'gemma4:12b' }));
    });
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
