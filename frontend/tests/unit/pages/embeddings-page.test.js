// frontend/tests/unit/pages/embeddings-page.test.js
import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import EmbeddingGeneration from '@/pages/embeddings';

// Mock the API functions
jest.mock('@/lib/api', () => ({
  embeddingApi: {
    generateEmbedding: jest.fn(),
  }
}));

import { embeddingApi } from '@/lib/api';

// Mock localStorage
Object.defineProperty(window, 'localStorage', {
  value: {
    getItem: jest.fn(() => 'test-auth-token'),
  },
  writable: true,
});

// Mock next/router
jest.mock('next/router', () => ({
  useRouter: () => ({
    query: {},
    push: jest.fn(),
  })
}));

// Mock alert
beforeAll(() => {
  jest.spyOn(window, 'alert').mockImplementation(() => {});
});

// Mock clipboard
Object.assign(navigator, {
  clipboard: {
    writeText: jest.fn(() => Promise.resolve()),
  }
});

describe('Embedding Generation Page', () => {
  const mockEmbedding = {
    embedding: Array.from({ length: 384 }, (_, i) => (Math.random() - 0.5) * 2)
  };

  beforeEach(() => {
    jest.clearAllMocks();

    // Mock successful API response
    embeddingApi.generateEmbedding.mockResolvedValue(mockEmbedding);
  });

  describe('Rendering', () => {
    test('renders page title', () => {
      render(<EmbeddingGeneration />);
      // "Generate Embedding" appears in both heading and button
      expect(screen.getByRole('heading', { name: /generate embedding/i })).toBeInTheDocument();
    });

    test('renders page description', () => {
      render(<EmbeddingGeneration />);
      expect(screen.getByText(/Generate vector embeddings from text or images/)).toBeInTheDocument();
    });

    test('renders text radio button', () => {
      render(<EmbeddingGeneration />);
      expect(screen.getByLabelText('Text')).toBeInTheDocument();
    });

    test('renders image radio button', () => {
      render(<EmbeddingGeneration />);
      expect(screen.getByLabelText('Image')).toBeInTheDocument();
    });

    test('renders model selector', () => {
      render(<EmbeddingGeneration />);
      expect(screen.getByText('Embedding Model')).toBeInTheDocument();
      expect(screen.getByRole('combobox')).toBeInTheDocument();
    });

    test('renders generate button', () => {
      render(<EmbeddingGeneration />);
      expect(screen.getByRole('button', { name: /generate embedding/i })).toBeInTheDocument();
    });

    test('defaults to text embedding type', () => {
      render(<EmbeddingGeneration />);
      expect(screen.getByLabelText('Text')).toBeChecked();
      expect(screen.getByLabelText('Image')).not.toBeChecked();
    });

    test('renders text input when text type selected', () => {
      render(<EmbeddingGeneration />);
      expect(screen.getByLabelText('Input Text')).toBeInTheDocument();
      expect(screen.getByPlaceholderText(/enter text to generate embedding/i)).toBeInTheDocument();
    });
  });

  describe('Embedding Type Selection', () => {
    test('shows text input when text type selected', () => {
      render(<EmbeddingGeneration />);

      // Text is default, should show text input
      expect(screen.getByLabelText('Input Text')).toBeInTheDocument();
      expect(screen.queryByText('Upload Image')).not.toBeInTheDocument();
    });

    test('shows image upload when image type selected', () => {
      render(<EmbeddingGeneration />);

      fireEvent.click(screen.getByLabelText('Image'));

      expect(screen.queryByLabelText('Input Text')).not.toBeInTheDocument();
      expect(screen.getByText('Upload Image')).toBeInTheDocument();
      expect(screen.getByText('Upload a file')).toBeInTheDocument();
    });

    test('switches back to text input', () => {
      render(<EmbeddingGeneration />);

      // Switch to image
      fireEvent.click(screen.getByLabelText('Image'));
      expect(screen.getByText('Upload Image')).toBeInTheDocument();

      // Switch back to text
      fireEvent.click(screen.getByLabelText('Text'));
      expect(screen.getByLabelText('Input Text')).toBeInTheDocument();
    });
  });

  describe('Model Selection', () => {
    test('shows text models for text type', () => {
      render(<EmbeddingGeneration />);

      const modelSelect = screen.getByRole('combobox');
      expect(modelSelect).toHaveValue('all-MiniLM-L6-v2');

      // Should show text models
      expect(screen.getByText(/all-MiniLM-L6-v2.*384D/)).toBeInTheDocument();
    });

    test('shows image models for image type', () => {
      render(<EmbeddingGeneration />);

      fireEvent.click(screen.getByLabelText('Image'));

      // Should show CLIP model for images
      expect(screen.getByText(/clip-ViT-B-32.*512D/)).toBeInTheDocument();
    });

    test('changes model selection', () => {
      render(<EmbeddingGeneration />);

      const modelSelect = screen.getByRole('combobox');
      fireEvent.change(modelSelect, { target: { value: 'all-mpnet-base-v2' } });

      expect(modelSelect).toHaveValue('all-mpnet-base-v2');
    });
  });

  describe('Text Embedding Generation', () => {
    test('calls API with text input', async () => {
      render(<EmbeddingGeneration />);

      const textInput = screen.getByLabelText('Input Text');
      fireEvent.change(textInput, { target: { value: 'Test text for embedding' } });

      fireEvent.click(screen.getByRole('button', { name: /generate embedding/i }));

      await waitFor(() => {
        expect(embeddingApi.generateEmbedding).toHaveBeenCalledWith({
          input: 'Test text for embedding',
          model: 'all-MiniLM-L6-v2',
          inputType: 'text'
        });
      });
    });

    test('shows loading state during generation', async () => {
      embeddingApi.generateEmbedding.mockImplementation(() => new Promise(() => {}));

      render(<EmbeddingGeneration />);

      const textInput = screen.getByLabelText('Input Text');
      fireEvent.change(textInput, { target: { value: 'Test text' } });

      fireEvent.click(screen.getByRole('button', { name: /generate embedding/i }));

      expect(screen.getByRole('button', { name: /generating/i })).toBeInTheDocument();
    });

    test('disables button during loading', async () => {
      embeddingApi.generateEmbedding.mockImplementation(() => new Promise(() => {}));

      render(<EmbeddingGeneration />);

      const textInput = screen.getByLabelText('Input Text');
      fireEvent.change(textInput, { target: { value: 'Test text' } });

      fireEvent.click(screen.getByRole('button', { name: /generate embedding/i }));

      expect(screen.getByRole('button', { name: /generating/i })).toBeDisabled();
    });

    test('disables button when no text entered', () => {
      render(<EmbeddingGeneration />);

      expect(screen.getByRole('button', { name: /generate embedding/i })).toBeDisabled();
    });

    test('enables button when text entered', () => {
      render(<EmbeddingGeneration />);

      const textInput = screen.getByLabelText('Input Text');
      fireEvent.change(textInput, { target: { value: 'Test text' } });

      expect(screen.getByRole('button', { name: /generate embedding/i })).not.toBeDisabled();
    });
  });

  describe('Image Embedding', () => {
    test('disables button when no image uploaded', () => {
      render(<EmbeddingGeneration />);

      fireEvent.click(screen.getByLabelText('Image'));

      expect(screen.getByRole('button', { name: /generate embedding/i })).toBeDisabled();
    });

    test('shows selected file name', () => {
      render(<EmbeddingGeneration />);

      fireEvent.click(screen.getByLabelText('Image'));

      const fileInput = screen.getByLabelText('Upload a file');
      const file = new File(['test'], 'test-image.png', { type: 'image/png' });

      fireEvent.change(fileInput, { target: { files: [file] } });

      expect(screen.getByText(/Selected: test-image.png/)).toBeInTheDocument();
    });

    test('calls API with image input', async () => {
      render(<EmbeddingGeneration />);

      fireEvent.click(screen.getByLabelText('Image'));

      const fileInput = screen.getByLabelText('Upload a file');
      const file = new File(['test'], 'test-image.png', { type: 'image/png' });

      fireEvent.change(fileInput, { target: { files: [file] } });

      // Change to CLIP model for image
      const modelSelect = screen.getByRole('combobox');
      fireEvent.change(modelSelect, { target: { value: 'clip-ViT-B-32' } });

      fireEvent.click(screen.getByRole('button', { name: /generate embedding/i }));

      await waitFor(() => {
        expect(embeddingApi.generateEmbedding).toHaveBeenCalledWith({
          input: 'test-image.png',
          model: 'clip-ViT-B-32',
          inputType: 'image'
        });
      });
    });
  });

  describe('Results Display', () => {
    test('displays generated embedding results', async () => {
      render(<EmbeddingGeneration />);

      const textInput = screen.getByLabelText('Input Text');
      fireEvent.change(textInput, { target: { value: 'Test text' } });

      fireEvent.click(screen.getByRole('button', { name: /generate embedding/i }));

      await waitFor(() => {
        expect(screen.getByText('Generated Embedding')).toBeInTheDocument();
      });
    });

    test('shows model name in results', async () => {
      render(<EmbeddingGeneration />);

      const textInput = screen.getByLabelText('Input Text');
      fireEvent.change(textInput, { target: { value: 'Test text' } });

      fireEvent.click(screen.getByRole('button', { name: /generate embedding/i }));

      await waitFor(() => {
        expect(screen.getByText(/Embedding generated with model: all-MiniLM-L6-v2/)).toBeInTheDocument();
      });
    });

    test('shows embedding dimensions', async () => {
      render(<EmbeddingGeneration />);

      const textInput = screen.getByLabelText('Input Text');
      fireEvent.change(textInput, { target: { value: 'Test text' } });

      fireEvent.click(screen.getByRole('button', { name: /generate embedding/i }));

      await waitFor(() => {
        expect(screen.getByText('Embedding Dimensions:')).toBeInTheDocument();
        expect(screen.getByText('384')).toBeInTheDocument();
      });
    });

    test('shows first 10 embedding values', async () => {
      render(<EmbeddingGeneration />);

      const textInput = screen.getByLabelText('Input Text');
      fireEvent.change(textInput, { target: { value: 'Test text' } });

      fireEvent.click(screen.getByRole('button', { name: /generate embedding/i }));

      await waitFor(() => {
        expect(screen.getByText(/Embedding Values \(first 10\)/)).toBeInTheDocument();
      });
    });

    test('shows copy to clipboard button', async () => {
      render(<EmbeddingGeneration />);

      const textInput = screen.getByLabelText('Input Text');
      fireEvent.change(textInput, { target: { value: 'Test text' } });

      fireEvent.click(screen.getByRole('button', { name: /generate embedding/i }));

      await waitFor(() => {
        expect(screen.getByRole('button', { name: /copy to clipboard/i })).toBeInTheDocument();
      });
    });

    test('shows download JSON button', async () => {
      render(<EmbeddingGeneration />);

      const textInput = screen.getByLabelText('Input Text');
      fireEvent.change(textInput, { target: { value: 'Test text' } });

      fireEvent.click(screen.getByRole('button', { name: /generate embedding/i }));

      await waitFor(() => {
        expect(screen.getByRole('button', { name: /download json/i })).toBeInTheDocument();
      });
    });
  });

  describe('Copy to Clipboard', () => {
    test('copies embedding to clipboard', async () => {
      render(<EmbeddingGeneration />);

      const textInput = screen.getByLabelText('Input Text');
      fireEvent.change(textInput, { target: { value: 'Test text' } });

      fireEvent.click(screen.getByRole('button', { name: /generate embedding/i }));

      await waitFor(() => {
        expect(screen.getByRole('button', { name: /copy to clipboard/i })).toBeInTheDocument();
      });

      fireEvent.click(screen.getByRole('button', { name: /copy to clipboard/i }));

      expect(navigator.clipboard.writeText).toHaveBeenCalledWith(JSON.stringify(mockEmbedding.embedding));
      expect(window.alert).toHaveBeenCalledWith('Embedding copied to clipboard!');
    });
  });

  describe('Error Handling', () => {
    test('shows alert on API error', async () => {
      embeddingApi.generateEmbedding.mockRejectedValue(new Error('Generation failed'));

      render(<EmbeddingGeneration />);

      const textInput = screen.getByLabelText('Input Text');
      fireEvent.change(textInput, { target: { value: 'Test text' } });

      fireEvent.click(screen.getByRole('button', { name: /generate embedding/i }));

      await waitFor(() => {
        expect(window.alert).toHaveBeenCalledWith(expect.stringContaining('Generation failed'));
      });
    });

    test('clears previous embedding on new generation', async () => {
      render(<EmbeddingGeneration />);

      // First generation
      const textInput = screen.getByLabelText('Input Text');
      fireEvent.change(textInput, { target: { value: 'Test text' } });

      fireEvent.click(screen.getByRole('button', { name: /generate embedding/i }));

      await waitFor(() => {
        expect(screen.getByText('Generated Embedding')).toBeInTheDocument();
      });

      // Second generation (starts loading, clears previous)
      embeddingApi.generateEmbedding.mockImplementation(() => new Promise(() => {}));

      fireEvent.click(screen.getByRole('button', { name: /generate embedding/i }));

      // Results should be cleared during loading
      expect(screen.queryByText('Generated Embedding')).not.toBeInTheDocument();
    });
  });

  describe('Model Info Display', () => {
    test('shows model description in options', () => {
      render(<EmbeddingGeneration />);

      // Multiple models may have similar descriptions
      const descriptions = screen.getAllByText(/Sentence transformer model/);
      expect(descriptions.length).toBeGreaterThan(0);
    });

    test('shows model output dimensions', () => {
      render(<EmbeddingGeneration />);

      expect(screen.getByText(/384D/)).toBeInTheDocument();
    });
  });
});
