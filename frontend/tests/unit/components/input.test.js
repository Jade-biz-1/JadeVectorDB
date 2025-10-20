// frontend/tests/unit/components/input.test.js
import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import { Input } from '@@/components/ui/input';

describe('Input Component', () => {
  test('renders input with correct type', () => {
    render(<Input type="text" data-testid="test-input" />);
    const input = screen.getByTestId('test-input');
    expect(input).toBeInTheDocument();
    expect(input.type).toBe('text');
  });

  test('applies default classes', () => {
    render(<Input data-testid="test-input" />);
    const input = screen.getByTestId('test-input');
    expect(input).toHaveClass('flex');
    expect(input).toHaveClass('h-10');
    expect(input).toHaveClass('w-full');
    expect(input).toHaveClass('rounded-md');
  });

  test('handles value changes', () => {
    const handleChange = jest.fn();
    render(
      <Input 
        data-testid="test-input" 
        onChange={handleChange}
        value="initial value"
      />
    );

    const input = screen.getByTestId('test-input');
    fireEvent.change(input, { target: { value: 'new value' } });

    expect(handleChange).toHaveBeenCalledTimes(1);
  });

  test('is disabled when disabled prop is true', () => {
    render(<Input disabled data-testid="test-input" />);
    const input = screen.getByTestId('test-input');
    expect(input).toBeDisabled();
  });

  test('renders with placeholder', () => {
    render(<Input placeholder="Enter text" data-testid="test-input" />);
    const input = screen.getByTestId('test-input');
    expect(input.placeholder).toBe('Enter text');
  });
});