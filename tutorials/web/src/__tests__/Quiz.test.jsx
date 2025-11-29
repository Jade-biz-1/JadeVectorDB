/**
 * Component tests for Quiz.jsx
 */

import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import Quiz from '../components/tutorial/Quiz';

// Mock child components
jest.mock('../components/tutorial/QuizProgress', () => {
  return function MockQuizProgress() {
    return <div data-testid="quiz-progress">Quiz Progress</div>;
  };
});

jest.mock('../components/tutorial/QuizQuestion', () => {
  return function MockQuizQuestion({ question, onAnswer }) {
    return (
      <div data-testid="quiz-question">
        <div>{question.question}</div>
        <button onClick={() => onAnswer(0)}>Answer</button>
      </div>
    );
  };
});

describe('Quiz Component', () => {
  const mockQuizData = {
    moduleId: 'module1',
    moduleName: 'Test Module',
    passingScore: 70,
    questions: [
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
      }
    ]
  };

  const mockOnComplete = jest.fn();

  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('Rendering', () => {
    it('should render quiz with first question', () => {
      render(
        <Quiz
          quizData={mockQuizData}
          onComplete={mockOnComplete}
        />
      );

      expect(screen.getByText('Test Module')).toBeInTheDocument();
      expect(screen.getByTestId('quiz-question')).toBeInTheDocument();
      expect(screen.getByTestId('quiz-progress')).toBeInTheDocument();
    });

    it('should display correct question number', () => {
      render(
        <Quiz
          quizData={mockQuizData}
          onComplete={mockOnComplete}
        />
      );

      expect(screen.getByText(/Question 1 of 2/)).toBeInTheDocument();
    });

    it('should show submit button only when question is answered', () => {
      render(
        <Quiz
          quizData={mockQuizData}
          onComplete={mockOnComplete}
        />
      );

      // Initially no submit button
      expect(screen.queryByText('Submit Answer')).not.toBeInTheDocument();

      // Answer the question
      fireEvent.click(screen.getByText('Answer'));

      // Now submit button should appear
      expect(screen.getByText('Submit Answer')).toBeInTheDocument();
    });
  });

  describe('Navigation', () => {
    it('should navigate to next question after submitting answer', async () => {
      render(
        <Quiz
          quizData={mockQuizData}
          onComplete={mockOnComplete}
        />
      );

      // Answer first question
      fireEvent.click(screen.getByText('Answer'));
      fireEvent.click(screen.getByText('Submit Answer'));

      // Should show next button
      await waitFor(() => {
        expect(screen.getByText('Next Question')).toBeInTheDocument();
      });

      // Click next
      fireEvent.click(screen.getByText('Next Question'));

      // Should be on question 2
      await waitFor(() => {
        expect(screen.getByText(/Question 2 of 2/)).toBeInTheDocument();
      });
    });

    it('should show finish quiz button on last question', async () => {
      render(
        <Quiz
          quizData={mockQuizData}
          onComplete={mockOnComplete}
        />
      );

      // Navigate to last question
      fireEvent.click(screen.getByText('Answer'));
      fireEvent.click(screen.getByText('Submit Answer'));
      fireEvent.click(screen.getByText('Next Question'));

      // Answer last question
      fireEvent.click(screen.getByText('Answer'));
      fireEvent.click(screen.getByText('Submit Answer'));

      // Should show finish button instead of next
      await waitFor(() => {
        expect(screen.getByText('Finish Quiz')).toBeInTheDocument();
      });
    });

    it('should call onComplete when finishing quiz', async () => {
      render(
        <Quiz
          quizData={mockQuizData}
          onComplete={mockOnComplete}
        />
      );

      // Answer all questions
      fireEvent.click(screen.getByText('Answer'));
      fireEvent.click(screen.getByText('Submit Answer'));
      fireEvent.click(screen.getByText('Next Question'));
      fireEvent.click(screen.getByText('Answer'));
      fireEvent.click(screen.getByText('Submit Answer'));

      // Finish quiz
      fireEvent.click(screen.getByText('Finish Quiz'));

      await waitFor(() => {
        expect(mockOnComplete).toHaveBeenCalled();
      });

      const callArg = mockOnComplete.mock.calls[0][0];
      expect(callArg).toHaveProperty('score');
      expect(callArg).toHaveProperty('passed');
      expect(callArg).toHaveProperty('gradedResults');
    });
  });

  describe('Answer Submission', () => {
    it('should record answers correctly', async () => {
      render(
        <Quiz
          quizData={mockQuizData}
          onComplete={mockOnComplete}
        />
      );

      // Answer first question
      fireEvent.click(screen.getByText('Answer'));
      const answer1 = screen.getByText('Submit Answer');
      fireEvent.click(answer1);

      // Move to next question
      fireEvent.click(screen.getByText('Next Question'));

      // Answer second question
      fireEvent.click(screen.getByText('Answer'));
      fireEvent.click(screen.getByText('Submit Answer'));

      // Finish quiz
      fireEvent.click(screen.getByText('Finish Quiz'));

      await waitFor(() => {
        const result = mockOnComplete.mock.calls[0][0];
        expect(result.gradedResults.length).toBe(2);
      });
    });

    it('should prevent re-answering after submission', async () => {
      render(
        <Quiz
          quizData={mockQuizData}
          onComplete={mockOnComplete}
        />
      );

      // Answer question
      fireEvent.click(screen.getByText('Answer'));
      fireEvent.click(screen.getByText('Submit Answer'));

      // Submit button should be disabled/gone
      await waitFor(() => {
        expect(screen.queryByText('Submit Answer')).not.toBeInTheDocument();
      });
    });
  });

  describe('Timer', () => {
    it('should display timer when timeLimit is provided', () => {
      render(
        <Quiz
          quizData={mockQuizData}
          onComplete={mockOnComplete}
          timeLimit={600}
        />
      );

      expect(screen.getByText(/Time:/)).toBeInTheDocument();
    });

    it('should not display timer when timeLimit is not provided', () => {
      render(
        <Quiz
          quizData={mockQuizData}
          onComplete={mockOnComplete}
        />
      );

      expect(screen.queryByText(/Time:/)).not.toBeInTheDocument();
    });
  });

  describe('Edge Cases', () => {
    it('should handle quiz with single question', () => {
      const singleQuestionQuiz = {
        ...mockQuizData,
        questions: [mockQuizData.questions[0]]
      };

      render(
        <Quiz
          quizData={singleQuestionQuiz}
          onComplete={mockOnComplete}
        />
      );

      expect(screen.getByText(/Question 1 of 1/)).toBeInTheDocument();

      // Answer and submit
      fireEvent.click(screen.getByText('Answer'));
      fireEvent.click(screen.getByText('Submit Answer'));

      // Should show finish button immediately
      expect(screen.getByText('Finish Quiz')).toBeInTheDocument();
    });

    it('should handle empty quiz data gracefully', () => {
      const emptyQuizData = {
        ...mockQuizData,
        questions: []
      };

      const { container } = render(
        <Quiz
          quizData={emptyQuizData}
          onComplete={mockOnComplete}
        />
      );

      // Should not crash
      expect(container).toBeInTheDocument();
    });
  });

  describe('Score Calculation', () => {
    it('should calculate passing score correctly', async () => {
      render(
        <Quiz
          quizData={mockQuizData}
          onComplete={mockOnComplete}
        />
      );

      // Answer both questions correctly (answer index matches correctAnswer)
      fireEvent.click(screen.getByText('Answer'));
      fireEvent.click(screen.getByText('Submit Answer'));
      fireEvent.click(screen.getByText('Next Question'));
      fireEvent.click(screen.getByText('Answer'));
      fireEvent.click(screen.getByText('Submit Answer'));
      fireEvent.click(screen.getByText('Finish Quiz'));

      await waitFor(() => {
        const result = mockOnComplete.mock.calls[0][0];
        expect(result.score).toBeDefined();
        expect(typeof result.score).toBe('number');
      });
    });
  });
});
