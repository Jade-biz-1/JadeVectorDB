import React from 'react';
import { render, screen } from '@testing-library/react';

// Global mocks: next/head, next/router, next/link via jest.config.js

import IntegrationGuide from '@/pages/integration';

describe('IntegrationGuide page', () => {
  it('renders the Integration Guide heading', () => {
    render(<IntegrationGuide />);
    expect(screen.getByRole('heading', { name: /integration guide/i })).toBeInTheDocument();
  });

  it('renders the How to Integrate section', () => {
    render(<IntegrationGuide />);
    expect(screen.getByRole('heading', { name: /how to integrate/i })).toBeInTheDocument();
  });

  it('renders REST API info', () => {
    render(<IntegrationGuide />);
    expect(screen.getByText(/REST API endpoints/i)).toBeInTheDocument();
  });

  it('renders a link to the API documentation', () => {
    render(<IntegrationGuide />);
    const apiDocLink = screen.getByRole('link', { name: /API documentation/i });
    expect(apiDocLink).toBeInTheDocument();
    expect(apiDocLink).toHaveAttribute('href', '/docs/api_documentation.md');
  });

  it('renders links to example code', () => {
    render(<IntegrationGuide />);
    expect(screen.getByRole('link', { name: /\/examples\/cli/i })).toBeInTheDocument();
    expect(screen.getByRole('link', { name: /\/examples\/frontend/i })).toBeInTheDocument();
  });
});
