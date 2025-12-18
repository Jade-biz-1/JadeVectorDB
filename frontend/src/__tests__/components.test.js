import React from 'react';
import { render, screen } from '@testing-library/react';
import '@testing-library/jest-dom';
import TutorialHeader from '../components/TutorialHeader';
import InstructionsPanel from '../components/InstructionsPanel';
import VisualDashboard from '../components/VisualDashboard';
import LivePreviewPanel from '../components/LivePreviewPanel';

// Mock the TutorialContext for testing
jest.mock('../contexts/TutorialContext', () => ({
  useTutorial: () => ({
    currentModule: 0,
    currentStep: 0,
    tutorialState: {
      modules: [
        {
          id: 0,
          title: "Getting Started",
          description: "Learn basic concepts",
          steps: 3,
          completed: 2
        }
      ]
    },
    setTutorialState: jest.fn(),
    setCurrentModule: jest.fn(),
    setCurrentStep: jest.fn()
  })
}));

// Mock the Monaco Editor
jest.mock('@monaco-editor/react', () => {
  return {
    __esModule: true,
    default: () => <div>Mock Code Editor</div>,
  };
});

// Mock D3
jest.mock('d3', () => ({
  select: () => ({
    selectAll: () => ({
      remove: jest.fn()
    }),
    append: () => ({
      attr: jest.fn().mockReturnThis(),
      style: jest.fn().mockReturnThis(),
      text: jest.fn().mockReturnThis()
    })
  })
}));

describe('Tutorial Components', () => {
  test('renders TutorialHeader component', () => {
    render(<TutorialHeader />);
    expect(screen.getByText('JadeVectorDB Interactive Tutorial')).toBeInTheDocument();
  });

  test('renders InstructionsPanel component', () => {
    render(<InstructionsPanel />);
    expect(screen.getByText('Tutorial Progress')).toBeInTheDocument();
  });

  test('renders VisualDashboard component', () => {
    render(<VisualDashboard />);
    expect(screen.getByText('Vector Space Visualization')).toBeInTheDocument();
  });

  test('renders LivePreviewPanel component', () => {
    render(<LivePreviewPanel />);
    expect(screen.getByText('Live Preview')).toBeInTheDocument();
  });
});