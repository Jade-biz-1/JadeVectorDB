import React from 'react';

/**
 * TrueFalseQuestion - True/False question component
 *
 * Simple binary choice question.
 */
const TrueFalseQuestion = ({
  question,
  answer,
  onChange,
  readOnly = false,
  showExplanation = false
}) => {
  /**
   * Handle answer selection
   */
  const handleSelect = (value) => {
    if (!readOnly) {
      onChange(value);
    }
  };

  /**
   * Check if option is selected
   */
  const isSelected = (value) => {
    return answer === value;
  };

  /**
   * Check if option is correct (for review mode)
   */
  const isCorrect = (value) => {
    return showExplanation && question.correctAnswer === value;
  };

  /**
   * Get option styling
   */
  const getOptionClassName = (value) => {
    const baseClasses = "flex-1 p-6 border-2 rounded-lg text-center font-semibold text-lg transition-all duration-200";
    const selected = isSelected(value);
    const correct = isCorrect(value);

    if (showExplanation) {
      // Review mode - show correct/incorrect
      if (correct) {
        return `${baseClasses} border-green-500 bg-green-50 text-green-900`;
      }
      if (selected && !correct) {
        return `${baseClasses} border-red-500 bg-red-50 text-red-900`;
      }
      return `${baseClasses} border-gray-300 bg-gray-50 text-gray-700`;
    }

    // Normal mode
    if (selected) {
      return `${baseClasses} border-blue-500 bg-blue-50 text-blue-900`;
    }

    if (readOnly) {
      return `${baseClasses} border-gray-300 bg-gray-50 text-gray-700 cursor-not-allowed`;
    }

    return `${baseClasses} border-gray-300 hover:border-gray-400 hover:bg-gray-50 cursor-pointer`;
  };

  /**
   * Get icon for option
   */
  const getIcon = (value) => {
    const selected = isSelected(value);
    const correct = isCorrect(value);

    if (showExplanation) {
      if (correct) {
        return '✓';
      }
      if (selected && !correct) {
        return '✗';
      }
      return '';
    }

    if (selected) {
      return value ? '✓' : '✗';
    }

    return value ? '⭕' : '⭕';
  };

  return (
    <div className="space-y-4">
      <div className="flex gap-4">
        {/* True button */}
        <button
          onClick={() => handleSelect(true)}
          disabled={readOnly}
          className={getOptionClassName(true)}
        >
          <div className="space-y-2">
            <div className="text-4xl">
              {getIcon(true) === '✓' ? '✅' : getIcon(true) === '✗' ? '❌' : '⭕'}
            </div>
            <div>TRUE</div>
            {showExplanation && isCorrect(true) && (
              <div className="text-sm text-green-700 mt-2">Correct Answer</div>
            )}
          </div>
        </button>

        {/* False button */}
        <button
          onClick={() => handleSelect(false)}
          disabled={readOnly}
          className={getOptionClassName(false)}
        >
          <div className="space-y-2">
            <div className="text-4xl">
              {getIcon(false) === '✓' ? '✅' : getIcon(false) === '✗' ? '❌' : '⭕'}
            </div>
            <div>FALSE</div>
            {showExplanation && isCorrect(false) && (
              <div className="text-sm text-green-700 mt-2">Correct Answer</div>
            )}
          </div>
        </button>
      </div>

      {/* Selection indicator */}
      {!readOnly && answer === undefined && (
        <p className="text-sm text-gray-500 text-center">
          Select TRUE or FALSE
        </p>
      )}

      {!readOnly && answer !== undefined && !showExplanation && (
        <p className="text-sm text-blue-600 text-center font-medium">
          You selected: {answer ? 'TRUE' : 'FALSE'}
        </p>
      )}
    </div>
  );
};

export default TrueFalseQuestion;
