import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import App from '@/App';
import { fetchBenchmark, fetchHealth, fetchModels } from '@/lib/api';
import type { HealthResponse, ModelsResponse } from '@/lib/types';

vi.mock('@/lib/api', () => ({
  fetchHealth: vi.fn(),
  fetchModels: vi.fn(),
  fetchBenchmark: vi.fn(),
  streamChat: vi.fn(() => () => undefined),
}));

const HEALTH: HealthResponse = {
  ollama_reachable: true,
  ollama_version: '0.6.0',
  collection_ready: true,
  indexed_chunks: 37,
  available_models: ['qwen3.5:9b'],
  configured_models: ['qwen3.5:9b'],
  missing_models: [],
};

const MODELS: ModelsResponse = {
  embedding_model: 'qwen3-embedding:0.6b',
  models: [{ name: 'qwen3.5:9b', role: 'primary', available: true }],
};

const healthMock = vi.mocked(fetchHealth);
const modelsMock = vi.mocked(fetchModels);

describe('App', () => {
  beforeEach(() => {
    vi.mocked(fetchBenchmark).mockResolvedValue({ available: false });
    healthMock.mockResolvedValue(HEALTH);
    modelsMock.mockResolvedValue(MODELS);
  });

  it('explains an unreachable backend instead of rendering an empty page', async () => {
    modelsMock.mockRejectedValue(new Error('/api/models failed with 502'));

    render(<App />);

    // Both the panel and the status strip have to say it — the strip used to
    // sit on "checking…" forever.
    expect(await screen.findAllByText('Sunucuya ulaşılamadı')).toHaveLength(2);
    expect(screen.getByRole('button', { name: 'Tekrar dene' })).toBeInTheDocument();
  });

  it('recovers when the backend comes back', async () => {
    const user = userEvent.setup();
    healthMock.mockRejectedValueOnce(new Error('down'));
    modelsMock.mockRejectedValueOnce(new Error('down'));

    render(<App />);
    await user.click(await screen.findByRole('button', { name: 'Tekrar dene' }));

    expect(await screen.findByText(/37 parça indeksli/)).toBeInTheDocument();
    expect(screen.queryAllByText('Sunucuya ulaşılamadı')).toHaveLength(0);
  });
});
