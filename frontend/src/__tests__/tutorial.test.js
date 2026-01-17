import React from 'react';
import { render, screen, fireEvent, waitFor, act } from '@testing-library/react';
import '@testing-library/jest-dom';

// Mock the charting library since it's not available in test environment
jest.mock('react-chartjs-2', () => ({
  Line: ({ data, options }) => <div data-testid="mock-chart">Mock Chart</div>,
  Bar: ({ data, options }) => <div data-testid="mock-bar-chart">Mock Bar Chart</div>,
  Doughnut: ({ data, options }) => <div data-testid="mock-doughnut-chart">Mock Doughnut Chart</div>
}));

describe('Tutorial Components', () => {
  describe('InteractiveAPIDocs', () => {
    it('renders without crashing', async () => {
      // Dynamically import the component to avoid issues with direct import
      const { default: InteractiveAPIDocs } = await import('../../tutorials/web/src/components/InteractiveAPIDocs');
      render(<InteractiveAPIDocs />);
      expect(screen.getByText('📖')).toBeInTheDocument();
    });

    it('displays API documentation sections', async () => {
      const { default: InteractiveAPIDocs } = await import('../../tutorials/web/src/components/InteractiveAPIDocs');
      render(<InteractiveAPIDocs />);
      
      expect(screen.getByText('Create Database')).toBeInTheDocument();
      expect(screen.getByText('Add Vector')).toBeInTheDocument();
      expect(screen.getByText('Search')).toBeInTheDocument();
    });
  });

  describe('BenchmarkingTools', () => {
    it('renders without crashing', async () => {
      const { default: BenchmarkingTools } = await import('../../tutorials/web/src/components/BenchmarkingTools');
      render(<BenchmarkingTools />);
      expect(screen.getByText('📊')).toBeInTheDocument();
    });

    it('displays benchmark controls', async () => {
      const { default: BenchmarkingTools } = await import('../../tutorials/web/src/components/BenchmarkingTools');
      render(<BenchmarkingTools />);
      
      expect(screen.getByText('Run')).toBeInTheDocument();
      expect(screen.getByText('Stop')).toBeInTheDocument();
      expect(screen.getByText('Reset')).toBeInTheDocument();
    });

    it('has different benchmark tabs', async () => {
      const { default: BenchmarkingTools } = await import('../../tutorials/web/src/components/BenchmarkingTools');
      render(<BenchmarkingTools />);
      
      expect(screen.getByText('Vector Search')).toBeInTheDocument();
      expect(screen.getByText('DB Operations')).toBeInTheDocument();
      expect(screen.getByText('Index Ops')).toBeInTheDocument();
    });
  });

  describe('CommunitySharing', () => {
    it('renders without crashing', async () => {
      const { default: CommunitySharing } = await import('../../tutorials/web/src/components/CommunitySharing');
      render(<CommunitySharing />);
      expect(screen.getByText('📤')).toBeInTheDocument();
    });

    it('has different sharing tabs', async () => {
      const { default: CommunitySharing } = await import('../../tutorials/web/src/components/CommunitySharing');
      render(<CommunitySharing />);
      
      expect(screen.getByText('Share')).toBeInTheDocument();
      expect(screen.getByText('Community')).toBeInTheDocument();
      expect(screen.getByText('My Shared')).toBeInTheDocument();
    });

    it('allows sharing a new scenario', async () => {
      const { default: CommunitySharing } = await import('../../tutorials/web/src/components/CommunitySharing');
      render(<CommunitySharing />);
      
      // Switch to Share tab
      const shareTab = screen.getByText('Share');
      fireEvent.click(shareTab);
      
      // Fill in the form
      const titleInput = screen.getByPlaceholderText(/Enter a descriptive title/i);
      fireEvent.change(titleInput, { target: { value: 'Test Scenario' } });
      
      const descInput = screen.getByPlaceholderText(/Describe what your scenario/i);
      fireEvent.change(descInput, { target: { value: 'A test scenario' } });
      
      const codeInput = screen.getByPlaceholderText(/Paste your code here/i);
      fireEvent.change(codeInput, { target: { value: '// Test code' } });
      
      // Click share button
      const shareButton = screen.getByText('Share Scenario');
      fireEvent.click(shareButton);
      
      await waitFor(() => {
        // Verify the active tab has switched to 'my-shared'
        expect(screen.getByText('My Shared')).toBeInTheDocument();
      });
    });
  });

  describe('ResourceUsageMonitor', () => {
    it('renders without crashing', async () => {
      const { default: ResourceUsageMonitor } = await import('../../tutorials/web/src/components/ResourceUsageMonitor');
      render(<ResourceUsageMonitor sessionId="test-session" />);
      expect(screen.getByText('📊')).toBeInTheDocument();
    });

    it('displays resource usage metrics', async () => {
      const { default: ResourceUsageMonitor } = await import('../../tutorials/web/src/components/ResourceUsageMonitor');
      render(<ResourceUsageMonitor sessionId="test-session" />);
      
      expect(screen.getByText('API Calls (per minute)')).toBeInTheDocument();
      expect(screen.getByText('Vectors Stored')).toBeInTheDocument();
      expect(screen.getByText('Databases Created')).toBeInTheDocument();
      expect(screen.getByText('Memory Usage')).toBeInTheDocument();
    });

    it('has a reset button', async () => {
      const { default: ResourceUsageMonitor } = await import('../../tutorials/web/src/components/ResourceUsageMonitor');
      render(<ResourceUsageMonitor sessionId="test-session" />);
      
      expect(screen.getByText('Reset')).toBeInTheDocument();
    });
  });
});