import React, { useState } from 'react';
import MultipleChoiceQuestion from './MultipleChoiceQuestion';
import TrueFalseQuestion from './TrueFalseQuestion';
import CodeChallengeQuestion from './CodeChallengeQuestion';

/**
 * QuizQuestion - Question router component
 *
 * Routes to the appropriate question type component and handles common features.
 */
const QuizQuestion = ({
  question,
  answer,
  onChange,
  questionNumber,
  totalQuestions,
  readOnly = false,
  showExplanation = false
}) => {
  const [showHints, setShowHints] = useState(false);
  const [currentHintLevel, setCurrentHintLevel] = useState(0);

  /**
   * Get the appropriate component for question type
   */
  const getQuestionComponent = () => {
    const commonProps = {
      question,
      answer,
      onChange,
      readOnly,
      showExplanation
    };

    switch (question.type) {
      case 'multiple-choice':
        return <MultipleChoiceQuestion {...commonProps} />;

      case 'true-false':
        return <TrueFalseQuestion {...commonProps} />;

      case 'code-challenge':
        return <CodeChallengeQuestion {...commonProps} />;

      case 'fill-blank':
        return <FillBlankQuestion {...commonProps} />;

      default:
        return (
          <div className="text-red-600">
            Unknown question type: {question.type}
          </div>
        );
    }
  };

  /**
   * Show next hint level
   */
  const handleShowHint = () => {
    if (question.hints && currentHintLevel < question.hints.length) {
      setCurrentHintLevel(currentHintLevel + 1);
      setShowHints(true);
    }
  };

  /**
   * Get difficulty badge color
   */
  const getDifficultyColor = () => {
    switch (question.difficulty?.toLowerCase()) {
      case 'easy':
        return 'bg-green-100 text-green-800 border-green-300';
      case 'medium':
        return 'bg-yellow-100 text-yellow-800 border-yellow-300';
      case 'hard':
        return 'bg-red-100 text-red-800 border-red-300';
      default:
        return 'bg-gray-100 text-gray-800 border-gray-300';
    }
  };

  return (
    <div className="space-y-4">
      {/* Question header */}
      <div className="flex items-start justify-between mb-4">
        <div className="flex-1">
          <div className="flex items-center gap-3 mb-2">
            <span className="text-2xl font-bold text-gray-800">
              Question {questionNumber}
            </span>
            {question.difficulty && (
              <span className={`px-3 py-1 rounded-full text-xs font-semibold border ${getDifficultyColor()}`}>
                {question.difficulty}
              </span>
            )}
            <span className="px-3 py-1 rounded-full text-xs font-semibold bg-blue-100 text-blue-800 border border-blue-300">
              {question.points} {question.points === 1 ? 'point' : 'points'}
            </span>
          </div>
        </div>
      </div>

      {/* Question text */}
      <div className="prose max-w-none">
        <p className="text-lg text-gray-800 leading-relaxed">
          {question.question}
        </p>
      </div>

      {/* Question component */}
      <div className="mt-6">
        {getQuestionComponent()}
      </div>

      {/* Hints section */}
      {!readOnly && !showExplanation && question.hints && question.hints.length > 0 && (
        <div className="mt-6">
          {!showHints ? (
            <button
              onClick={() => setShowHints(true)}
              className="text-sm text-blue-600 hover:text-blue-700 font-medium"
            >
              💡 Need a hint?
            </button>
          ) : (
            <div className="space-y-3">
              {question.hints.slice(0, currentHintLevel).map((hint, index) => (
                <div
                  key={index}
                  className="p-4 bg-yellow-50 border border-yellow-200 rounded-lg"
                >
                  <div className="flex items-start gap-2">
                    <span className="text-yellow-600 font-semibold">💡</span>
                    <div className="flex-1">
                      <p className="text-sm text-yellow-900">
                        <strong>Hint {index + 1}:</strong> {hint}
                      </p>
                    </div>
                  </div>
                </div>
              ))}

              {currentHintLevel < question.hints.length && (
                <button
                  onClick={handleShowHint}
                  className="text-sm text-blue-600 hover:text-blue-700 font-medium"
                >
                  Show another hint ({question.hints.length - currentHintLevel} remaining)
                </button>
              )}
            </div>
          )}
        </div>
      )}

      {/* Explanation (shown after submission) */}
      {showExplanation && question.explanation && (
        <div className="mt-6 p-4 bg-blue-50 border border-blue-200 rounded-lg">
          <div className="flex items-start gap-2">
            <span className="text-blue-600 text-xl">📖</span>
            <div className="flex-1">
              <h4 className="font-semibold text-blue-900 mb-2">Explanation:</h4>
              <p className="text-sm text-blue-800">{question.explanation}</p>
            </div>
          </div>
        </div>
      )}

      {/* Progress indicator */}
      <div className="mt-6 pt-4 border-t border-gray-200 text-sm text-gray-500">
        Question {questionNumber} of {totalQuestions}
      </div>
    </div>
  );
};

/**
 * FillBlankQuestion - Fill in the blank question component
 */
const FillBlankQuestion = ({ question, answer, onChange, readOnly }) => {
  return (
    <div className="space-y-4">
      <input
        type="text"
        value={answer || ''}
        onChange={(e) => !readOnly && onChange(e.target.value)}
        readOnly={readOnly}
        placeholder="Type your answer here..."
        className="w-full px-4 py-3 border-2 border-gray-300 rounded-lg focus:border-blue-500 focus:outline-none text-lg"
      />
    </div>
  );
};

export default QuizQuestion;
