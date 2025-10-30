// frontend/tests/unit/components/select.test.js
import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import { Select } from '@/components/ui/select';

describe('Select Component', () => {
  test('renders select with options', () => {
    render(
      <Select data-testid="test-select">
        <option value="option1">Option 1</option>
        <option value="option2">Option 2</option>
      </Select>
    );

    const select = screen.getByTestId('test-select');
    expect(select).toBeInTheDocument();
    
    const options = screen.getAllByRole('option');
    expect(options).toHaveLength(2);
    expect(options[0]).toHaveTextContent('Option 1');
    expect(options[1]).toHaveTextContent('Option 2');
  });

  test('applies default classes', () => {
    render(
      <Select data-testid="test-select">
        <option value="test">Test</option>
      </Select>
    );

    const select = screen.getByTestId('test-select');
    expect(select).toHaveClass('flex');
    expect(select).toHaveClass('h-10');
    expect(select).toHaveClass('w-full');
    expect(select).toHaveClass('rounded-md');
  });

  test('handles change events', () => {
    const handleChange = jest.fn();
    render(
      <Select data-testid="test-select" onChange={handleChange}>
        <option value="option1">Option 1</option>
        <option value="option2">Option 2</option>
      </Select>
    );

    const select = screen.getByTestId('test-select');
    fireEvent.change(select, { target: { value: 'option2' } });

    expect(handleChange).toHaveBeenCalledTimes(1);
    expect(select.value).toBe('option2');
  });

  test('is disabled when disabled prop is true', () => {
    render(
      <Select disabled data-testid="test-select">
        <option value="test">Test</option>
      </Select>
    );

    const select = screen.getByTestId('test-select');
    expect(select).toBeDisabled();
  });
});