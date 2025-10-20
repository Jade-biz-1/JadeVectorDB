// frontend/tests/unit/components/alert.test.js
import React from 'react';
import { render, screen } from '@testing-library/react';
import { Alert, AlertTitle, AlertDescription } from '@@/components/ui/alert';

describe('Alert Components', () => {
  test('renders Alert with correct structure', () => {
    render(
      <Alert>
        <AlertTitle>Test Title</AlertTitle>
        <AlertDescription>Test Description</AlertDescription>
      </Alert>
    );

    expect(screen.getByText('Test Title')).toBeInTheDocument();
    expect(screen.getByText('Test Description')).toBeInTheDocument();
  });

  test('Alert applies base classes', () => {
    render(<Alert data-testid="test-alert">Test Alert</Alert>);
    const alert = screen.getByTestId('test-alert');
    expect(alert).toHaveClass('relative');
    expect(alert).toHaveClass('w-full');
    expect(alert).toHaveClass('rounded-lg');
    expect(alert).toHaveClass('border');
    expect(alert).toHaveClass('p-4');
  });

  test('AlertTitle renders with correct element', () => {
    render(
      <Alert>
        <AlertTitle>Test Title</AlertTitle>
      </Alert>
    );

    const title = screen.getByText('Test Title');
    expect(title.tagName).toBe('H5');
    expect(title).toHaveClass('mb-1');
    expect(title).toHaveClass('font-medium');
  });

  test('AlertDescription renders with correct element', () => {
    render(
      <Alert>
        <AlertDescription>Test Description</AlertDescription>
      </Alert>
    );

    const description = screen.getByText('Test Description');
    expect(description.tagName).toBe('DIV');
    expect(description).toHaveClass('text-sm');
  });

  test('applies different variants', () => {
    const { rerender } = render(<Alert variant="default" data-testid="test-alert">Test</Alert>);
    let alert = screen.getByTestId('test-alert');
    expect(alert).toHaveClass('border-gray-200');
    expect(alert).toHaveClass('bg-white');

    rerender(<Alert variant="destructive" data-testid="test-alert">Test</Alert>);
    alert = screen.getByTestId('test-alert');
    expect(alert).toHaveClass('border-red-200/50');
    expect(alert).toHaveClass('[&>svg]:text-destructive');
  });
});