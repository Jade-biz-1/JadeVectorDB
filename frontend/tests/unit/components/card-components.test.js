// frontend/tests/unit/components/card-components.test.js
import React from 'react';
import { render, screen } from '@testing-library/react';
import {
  Card,
  CardHeader,
  CardFooter,
  CardTitle,
  CardDescription,
  CardContent
} from '@/components/ui/card';

describe('Card Components Suite', () => {
  test('renders Card with proper structure', () => {
    render(
      <Card data-testid="test-card">
        <CardHeader>
          <CardTitle>Test Title</CardTitle>
          <CardDescription>Test Description</CardDescription>
        </CardHeader>
        <CardContent>
          <p>Card content goes here</p>
        </CardContent>
        <CardFooter>
          <p>Card footer content</p>
        </CardFooter>
      </Card>
    );

    const card = screen.getByTestId('test-card');
    expect(card).toBeInTheDocument();
    // Updated to match actual Card component classes
    expect(card).toHaveClass('rounded-lg');
    expect(card).toHaveClass('border');
    expect(card).toHaveClass('bg-card');
    expect(card).toHaveClass('shadow-sm');
  });

  test('renders CardTitle with correct styling', () => {
    render(<CardTitle>My Card Title</CardTitle>);

    const title = screen.getByText('My Card Title');
    expect(title).toBeInTheDocument();
    expect(title.tagName).toBe('H3');
    expect(title).toHaveClass('text-2xl');
    expect(title).toHaveClass('font-semibold');
  });

  test('renders CardDescription with correct styling', () => {
    render(<CardDescription>This is a description</CardDescription>);

    const description = screen.getByText('This is a description');
    expect(description).toBeInTheDocument();
    expect(description.tagName).toBe('P');
    expect(description).toHaveClass('text-sm');
    expect(description).toHaveClass('text-muted-foreground');
  });

  test('renders CardContent with correct styling', () => {
    render(
      <CardContent>
        <p>Content inside card</p>
      </CardContent>
    );

    const content = screen.getByText('Content inside card');
    expect(content).toBeInTheDocument();
    expect(content.parentElement).toHaveClass('p-6');
    expect(content.parentElement).toHaveClass('pt-0');
  });

  test('renders CardHeader with proper structure', () => {
    render(
      <CardHeader>
        <CardTitle>Header Title</CardTitle>
      </CardHeader>
    );

    const header = screen.getByRole('heading', { name: /header title/i });
    expect(header).toBeInTheDocument();
    expect(header.parentElement).toHaveClass('flex');
    expect(header.parentElement).toHaveClass('flex-col');
    expect(header.parentElement).toHaveClass('space-y-1.5');
    expect(header.parentElement).toHaveClass('p-6');
  });

  test('renders CardFooter with proper structure', () => {
    render(
      <CardFooter>
        <button>Footer Button</button>
      </CardFooter>
    );

    const button = screen.getByText('Footer Button');
    expect(button).toBeInTheDocument();
    expect(button.parentElement).toHaveClass('flex');
    expect(button.parentElement).toHaveClass('items-center');
    expect(button.parentElement).toHaveClass('p-6');
    expect(button.parentElement).toHaveClass('pt-0');
  });

  test('allows custom class names to be passed through', () => {
    render(
      <Card className="custom-class another-class" data-testid="custom-card">
        <CardHeader>
          <CardTitle>Title</CardTitle>
        </CardHeader>
        <CardContent>Content</CardContent>
      </Card>
    );

    // Use data-testid instead of getByRole('group') since Card is a plain div
    const card = screen.getByTestId('custom-card');
    expect(card).toHaveClass('custom-class');
    expect(card).toHaveClass('another-class');
    // Also should have the default classes (matching actual implementation)
    expect(card).toHaveClass('rounded-lg');
    expect(card).toHaveClass('border');
    expect(card).toHaveClass('bg-card');
    expect(card).toHaveClass('shadow-sm');
  });

  test('handles nested elements correctly', () => {
    render(
      <Card>
        <CardHeader>
          <CardTitle>Title with <strong>emphasis</strong></CardTitle>
          <CardDescription>Description with <em>italics</em></CardDescription>
        </CardHeader>
        <CardContent>
          <ul>
            <li>List item 1</li>
            <li>List item 2</li>
          </ul>
        </CardContent>
      </Card>
    );

    expect(screen.getByText('emphasis')).toBeInTheDocument();
    expect(screen.getByText('italics')).toBeInTheDocument();
    expect(screen.getByText('List item 1')).toBeInTheDocument();
    expect(screen.getByText('List item 2')).toBeInTheDocument();
  });
});
