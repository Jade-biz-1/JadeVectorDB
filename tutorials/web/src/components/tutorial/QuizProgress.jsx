import React from 'react';

/**
 * QuizProgress - Progress indicator for quiz
 *
 * Shows progress bar and question navigation.
 */
const QuizProgress = ({
  currentQuestion,
  totalQuestions,
  answeredCount,
  onQuestionClick,
  questions,
  answers
}) => {
  const progressPercentage = (currentQuestion / totalQuestions) * 100;
  const answeredPercentage = (answeredCount / totalQuestions) * 100;

  /**
   * Get status for a question
   */
  const getQuestionStatus = (index) => {
    const question = questions[index];
    const hasAnswer = answers && answers[question.id] !== undefined;

    if (index === currentQuestion - 1) {
      return 'current';
    }
    if (hasAnswer) {
      return 'answered';
    }
    return 'unanswered';
  };

  /**
   * Get color class for question status
   */
  const getStatusColor = (status) => {
    switch (status) {
      case 'current':
        return 'bg-blue-600 border-blue-600 text-white';
      case 'answered':
        return 'bg-green-500 border-green-500 text-white';
      case 'unanswered':
        return 'bg-white border-gray-300 text-gray-600 hover:border-gray-400';
      default:
        return 'bg-white border-gray-300 text-gray-600';
    }
  };

  return (
    <div className="space-y-4">
      {/* Progress bar */}
      <div>
        <div className="flex justify-between text-sm text-gray-600 mb-2">
          <span>Progress</span>
          <span>{answeredCount} of {totalQuestions} answered ({Math.round(answeredPercentage)}%)</span>
        </div>
        <div className="w-full bg-gray-200 rounded-full h-3 overflow-hidden">
          <div
            className="bg-blue-600 h-full transition-all duration-300 ease-out"
            style={{ width: `${progressPercentage}%` }}
          >
            <div
              className="bg-green-500 h-full transition-all duration-300 ease-out"
              style={{ width: `${(answeredCount / totalQuestions) * 100}%` }}
            />
          </div>
        </div>
      </div>

      {/* Question dots/numbers */}
      {onQuestionClick && (
        <div className="flex flex-wrap gap-2">
          {questions.map((question, index) => {
            const status = getQuestionStatus(index);
            const isClickable = onQuestionClick !== null;

            return (
              <button
                key={question.id}
                onClick={() => isClickable && onQuestionClick(index)}
                disabled={!isClickable}
                className={`
                  w-10 h-10 rounded-full border-2 font-semibold text-sm
                  transition-all duration-200
                  ${getStatusColor(status)}
                  ${isClickable ? 'cursor-pointer' : 'cursor-default'}
                  ${status === 'current' ? 'ring-2 ring-blue-300 ring-offset-2' : ''}
                `}
                title={`Question ${index + 1}${status === 'answered' ? ' (Answered)' : ''}`}
              >
                {index + 1}
              </button>
            );
          })}
        </div>
      )}

      {/* Legend */}
      <div className="flex flex-wrap gap-4 text-sm">
        <div className="flex items-center gap-2">
          <div className="w-4 h-4 rounded-full bg-blue-600"></div>
          <span className="text-gray-600">Current</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-4 h-4 rounded-full bg-green-500"></div>
          <span className="text-gray-600">Answered</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-4 h-4 rounded-full bg-white border-2 border-gray-300"></div>
          <span className="text-gray-600">Unanswered</span>
        </div>
      </div>
    </div>
  );
};

export default QuizProgress;
