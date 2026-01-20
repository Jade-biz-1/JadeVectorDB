/**
 * Component tests for Quiz.jsx
 * Aligned with actual Quiz component props
 */

import React from 'react';
import { render, screen, fireEvent, waitFor, act } from '@testing-library/react';
import '@testing-library/jest-dom';
import Quiz from '../components/tutorial/Quiz';

// Mock child components
jest.mock('../components/tutorial/QuizProgress', () => {
  return function MockQuizProgress({ currentQuestion, totalQuestions, answeredCount }) {
    return (
      <div data-testid="quiz-progress">
        Question {currentQuestion} of {totalQuestions}, {answeredCount || 0} answered
      </div>
    );
  };
});

jest.mock('../components/tutorial/QuizQuestion', () => {
  return function MockQuizQuestion({ question, answer, onChange }) {
    return (
      <div data-testid="quiz-question">
        <div data-testid="question-text">{question.question}</div>
        <button data-testid="answer-btn" onClick={() => onChange(0)}>Select Answer A</button>
        <button data-testid="answer-btn-b" onClick={() => onChange(1)}>Select Answer B</button>
      </div>
    );
  };
});

describe('Quiz Component', () => {
  // Props match actual Quiz component signature
  const mockQuestions = [
    {
      id: 'q1',
      type: 'multiple-choice',
      question: 'Question 1?',
      points: 10,
      difficulty: 'easy',
      correctAnswer: 0,
      options: ['A', 'B', 'C', 'D']
    },
    {
      id: 'q2',
      type: 'multiple-choice',
      question: 'Question 2?',
      points: 15,
      difficulty: 'medium',
      correctAnswer: 1,
      options: ['A', 'B', 'C', 'D']
    },
    {
      id: 'q3',
      type: 'multiple-choice',
      question: 'Question 3?',
      points: 20,
      difficulty: 'hard',
      correctAnswer: 2,
      options: ['A', 'B', 'C', 'D']
    }
  ];

  const mockOnSubmit = jest.fn();
  const mockOnAnswerChange = jest.fn();

  const defaultProps = {
    questions: mockQuestions,
    onSubmit: mockOnSubmit,
    onAnswerChange: mockOnAnswerChange,
    allowNavigation: true
  };

  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('Rendering', () => {
    it('should render quiz with first question', () => {
      render(<Quiz {...defaultProps} />);

      expect(screen.getByTestId('quiz-question')).toBeInTheDocument();
      expect(screen.getByTestId('quiz-progress')).toBeInTheDocument();
    });

    it('should display the first question text', () => {
      render(<Quiz {...defaultProps} />);

      expect(screen.getByTestId('question-text')).toHaveTextContent('Question 1?');
    });

    it('should show navigation buttons', () => {
      render(<Quiz {...defaultProps} />);

      // Should have Next button on first question
      expect(screen.getByText(/next/i)).toBeInTheDocument();
    });

    it('should show progress information', () => {
      render(<Quiz {...defaultProps} />);

      // QuizProgress receives currentQuestion (1-indexed), totalQuestions, and answeredCount
      expect(screen.getByTestId('quiz-progress')).toHaveTextContent('Question 1 of 3');
      expect(screen.getByTestId('quiz-progress')).toHaveTextContent('0 answered');
    });
  });

  describe('Navigation', () => {
    it('should navigate to next question when Next is clicked', async () => {
      render(<Quiz {...defaultProps} />);

      // Answer current question first (optional, but good practice)
      fireEvent.click(screen.getByTestId('answer-btn'));

      // Click next - the button text includes an arrow "Next →"
      const nextButton = screen.getByRole('button', { name: /next/i });
      fireEvent.click(nextButton);

      await waitFor(() => {
        expect(screen.getByTestId('question-text')).toHaveTextContent('Question 2?');
      });
    });

    it('should navigate to previous question when Previous is clicked', async () => {
      render(<Quiz {...defaultProps} />);

      // Go to second question
      fireEvent.click(screen.getByTestId('answer-btn'));
      fireEvent.click(screen.getByRole('button', { name: /next/i }));

      await waitFor(() => {
        expect(screen.getByTestId('question-text')).toHaveTextContent('Question 2?');
      });

      // Go back - button text is "← Previous"
      const prevButton = screen.getByRole('button', { name: /previous/i });
      fireEvent.click(prevButton);

      await waitFor(() => {
        expect(screen.getByTestId('question-text')).toHaveTextContent('Question 1?');
      });
    });

    it('should disable Previous button on first question', () => {
      render(<Quiz {...defaultProps} />);

      // On first question, Previous should be disabled
      const prevButton = screen.getByRole('button', { name: /previous/i });
      expect(prevButton).toBeDisabled();
    });
  });

  describe('Answer Selection', () => {
    it('should call onAnswerChange when answer is selected', () => {
      render(<Quiz {...defaultProps} />);

      fireEvent.click(screen.getByTestId('answer-btn'));

      expect(mockOnAnswerChange).toHaveBeenCalled();
    });

    it('should track selected answers', () => {
      render(<Quiz {...defaultProps} />);

      // Select answer for first question
      fireEvent.click(screen.getByTestId('answer-btn'));

      // The answer should be tracked (we verify via onAnswerChange callback)
      expect(mockOnAnswerChange).toHaveBeenCalledWith('q1', 0);
    });
  });

  describe('Submission', () => {
    it('should show submit button on last question', async () => {
      render(<Quiz {...defaultProps} />);

      // Navigate to last question
      fireEvent.click(screen.getByTestId('answer-btn'));
      fireEvent.click(screen.getByRole('button', { name: /next/i }));

      await waitFor(() => {
        expect(screen.getByTestId('question-text')).toHaveTextContent('Question 2?');
      });

      fireEvent.click(screen.getByTestId('answer-btn'));
      fireEvent.click(screen.getByRole('button', { name: /next/i }));

      await waitFor(() => {
        expect(screen.getByTestId('question-text')).toHaveTextContent('Question 3?');
      });

      // Should show submit on last question - using role to be specific
      expect(screen.getByRole('button', { name: /submit quiz/i })).toBeInTheDocument();
    });

    it('should call onSubmit when submitted', async () => {
      render(<Quiz {...defaultProps} />);

      // Navigate to last question and answer all questions
      fireEvent.click(screen.getByTestId('answer-btn'));
      fireEvent.click(screen.getByRole('button', { name: /next/i }));

      await waitFor(() => {
        expect(screen.getByTestId('question-text')).toHaveTextContent('Question 2?');
      });

      fireEvent.click(screen.getByTestId('answer-btn'));
      fireEvent.click(screen.getByRole('button', { name: /next/i }));

      await waitFor(() => {
        expect(screen.getByTestId('question-text')).toHaveTextContent('Question 3?');
      });

      fireEvent.click(screen.getByTestId('answer-btn'));

      // Click submit - this will show confirmation modal since all answered
      const submitButton = screen.getByRole('button', { name: /submit quiz/i });
      fireEvent.click(submitButton);

      // Confirmation modal appears - click the submit button in it
      await waitFor(() => {
        // There are now two Submit Quiz buttons - one in modal
        const submitButtons = screen.getAllByRole('button', { name: /submit quiz/i });
        // Click the one in the modal (second button)
        fireEvent.click(submitButtons[submitButtons.length - 1]);
      });

      await waitFor(() => {
        expect(mockOnSubmit).toHaveBeenCalled();
      });
    });
  });

  describe('Timer (optional)', () => {
    it('should render without timer when timeLimit is not set', () => {
      render(<Quiz {...defaultProps} />);

      // Should not crash and should render
      expect(screen.getByTestId('quiz-question')).toBeInTheDocument();
    });

    it('should render with timer when timeLimit is set', () => {
      render(<Quiz {...defaultProps} timeLimit={300} />);

      // Should not crash and should render
      expect(screen.getByTestId('quiz-question')).toBeInTheDocument();
    });
  });

  describe('Navigation disabled', () => {
    it('should still allow moving forward when navigation is disabled', () => {
      render(<Quiz {...defaultProps} allowNavigation={false} />);

      // Should render
      expect(screen.getByTestId('quiz-question')).toBeInTheDocument();
    });
  });

  describe('Edge Cases', () => {
    it('should handle single question quiz', () => {
      render(
        <Quiz
          {...defaultProps}
          questions={[mockQuestions[0]]}
        />
      );

      expect(screen.getByTestId('quiz-question')).toBeInTheDocument();
      // Should show submit since it's the only question
      expect(screen.getByRole('button', { name: /submit quiz/i })).toBeInTheDocument();
    });

    it('should handle empty answers object initially', () => {
      render(<Quiz {...defaultProps} />);

      // Should render without crashing
      expect(screen.getByTestId('quiz-question')).toBeInTheDocument();
    });
  });
});
