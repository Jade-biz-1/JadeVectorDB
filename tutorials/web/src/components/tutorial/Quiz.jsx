import React, { useState, useEffect } from 'react';
import QuizQuestion from './QuizQuestion';
import QuizProgress from './QuizProgress';

/**
 * Quiz - Quiz presentation and navigation component
 *
 * Handles question display, navigation, and submission.
 */
const Quiz = ({
  questions,
  onSubmit,
  onAnswerChange,
  allowNavigation = true,
  timeLimit = null
}) => {
  const [currentQuestionIndex, setCurrentQuestionIndex] = useState(0);
  const [answers, setAnswers] = useState({});
  const [timeRemaining, setTimeRemaining] = useState(timeLimit);
  const [showSubmitConfirm, setShowSubmitConfirm] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);

  const currentQuestion = questions[currentQuestionIndex];
  const isLastQuestion = currentQuestionIndex === questions.length - 1;
  const isFirstQuestion = currentQuestionIndex === 0;

  // Timer effect
  useEffect(() => {
    if (timeLimit && timeRemaining > 0) {
      const timer = setInterval(() => {
        setTimeRemaining(prev => {
          if (prev <= 1) {
            // Time's up - auto submit
            handleSubmit();
            return 0;
          }
          return prev - 1;
        });
      }, 1000);

      return () => clearInterval(timer);
    }
  }, [timeLimit, timeRemaining]);

  /**
   * Handle answer change for current question
   */
  const handleAnswerChange = (answer) => {
    const newAnswers = {
      ...answers,
      [currentQuestion.id]: answer
    };
    setAnswers(newAnswers);

    // Notify parent component
    if (onAnswerChange) {
      onAnswerChange(currentQuestion.id, answer);
    }
  };

  /**
   * Navigate to next question
   */
  const handleNext = () => {
    if (!isLastQuestion) {
      setCurrentQuestionIndex(prev => prev + 1);
    }
  };

  /**
   * Navigate to previous question
   */
  const handlePrevious = () => {
    if (!isFirstQuestion && allowNavigation) {
      setCurrentQuestionIndex(prev => prev - 1);
    }
  };

  /**
   * Jump to specific question
   */
  const handleGoToQuestion = (index) => {
    if (allowNavigation && index >= 0 && index < questions.length) {
      setCurrentQuestionIndex(index);
    }
  };

  /**
   * Check how many questions are answered
   */
  const getAnsweredCount = () => {
    return Object.keys(answers).length;
  };

  /**
   * Check if all questions are answered
   */
  const isAllAnswered = () => {
    return getAnsweredCount() === questions.length;
  };

  /**
   * Handle quiz submission
   */
  const handleSubmit = () => {
    if (isSubmitting) return;

    setIsSubmitting(true);
    setShowSubmitConfirm(false);

    // Submit answers
    if (onSubmit) {
      onSubmit(answers);
    }
  };

  /**
   * Show submit confirmation dialog
   */
  const handleSubmitClick = () => {
    if (!isAllAnswered()) {
      const unanswered = questions.length - getAnsweredCount();
      const confirmMessage = `You have ${unanswered} unanswered question${unanswered > 1 ? 's' : ''}. Submit anyway?`;
      if (window.confirm(confirmMessage)) {
        handleSubmit();
      }
    } else {
      setShowSubmitConfirm(true);
    }
  };

  /**
   * Format time remaining
   */
  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  return (
    <div className="space-y-6">
      {/* Timer */}
      {timeLimit && (
        <div className={`p-4 rounded-lg ${timeRemaining < 60 ? 'bg-red-50 border border-red-300' : 'bg-blue-50 border border-blue-300'}`}>
          <div className="flex items-center justify-between">
            <span className="font-semibold">Time Remaining:</span>
            <span className={`text-xl font-bold ${timeRemaining < 60 ? 'text-red-600' : 'text-blue-600'}`}>
              {formatTime(timeRemaining)}
            </span>
          </div>
        </div>
      )}

      {/* Progress bar */}
      <QuizProgress
        currentQuestion={currentQuestionIndex + 1}
        totalQuestions={questions.length}
        answeredCount={getAnsweredCount()}
        onQuestionClick={allowNavigation ? handleGoToQuestion : null}
        questions={questions}
        answers={answers}
      />

      {/* Question */}
      <div className="bg-white rounded-lg shadow-lg p-6">
        <QuizQuestion
          question={currentQuestion}
          answer={answers[currentQuestion.id]}
          onChange={handleAnswerChange}
          questionNumber={currentQuestionIndex + 1}
          totalQuestions={questions.length}
        />
      </div>

      {/* Navigation */}
      <div className="flex justify-between items-center">
        <button
          onClick={handlePrevious}
          disabled={isFirstQuestion || !allowNavigation}
          className={`px-6 py-3 rounded-lg font-semibold ${
            isFirstQuestion || !allowNavigation
              ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
              : 'bg-gray-600 text-white hover:bg-gray-700'
          }`}
        >
          ← Previous
        </button>

        <div className="text-sm text-gray-600">
          Question {currentQuestionIndex + 1} of {questions.length}
          {' • '}
          {getAnsweredCount()} answered
        </div>

        {!isLastQuestion ? (
          <button
            onClick={handleNext}
            className="px-6 py-3 bg-blue-600 text-white rounded-lg font-semibold hover:bg-blue-700"
          >
            Next →
          </button>
        ) : (
          <button
            onClick={handleSubmitClick}
            disabled={isSubmitting}
            className={`px-6 py-3 rounded-lg font-semibold ${
              isSubmitting
                ? 'bg-gray-400 cursor-not-allowed'
                : 'bg-green-600 hover:bg-green-700'
            } text-white`}
          >
            {isSubmitting ? 'Submitting...' : 'Submit Quiz'}
          </button>
        )}
      </div>

      {/* Submit confirmation modal */}
      {showSubmitConfirm && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 max-w-md mx-4">
            <h3 className="text-xl font-bold mb-4">Submit Quiz?</h3>
            <p className="text-gray-600 mb-6">
              You have answered all {questions.length} questions. Are you ready to submit your quiz?
            </p>
            <div className="flex justify-end space-x-4">
              <button
                onClick={() => setShowSubmitConfirm(false)}
                className="px-4 py-2 bg-gray-300 text-gray-700 rounded hover:bg-gray-400"
              >
                Review Answers
              </button>
              <button
                onClick={handleSubmit}
                className="px-4 py-2 bg-green-600 text-white rounded hover:bg-green-700"
              >
                Submit Quiz
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Navigation hint */}
      {allowNavigation && questions.length > 1 && (
        <div className="text-center text-sm text-gray-500 mt-4">
          💡 Tip: You can review and change your answers before submitting
        </div>
      )}
    </div>
  );
};

export default Quiz;
