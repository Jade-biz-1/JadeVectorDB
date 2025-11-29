import React from 'react';

/**
 * MultipleChoiceQuestion - Multiple choice question component
 *
 * Supports both single-answer and multiple-answer questions.
 */
const MultipleChoiceQuestion = ({
  question,
  answer,
  onChange,
  readOnly = false,
  showExplanation = false
}) => {
  const isMultipleAnswer = question.multipleAnswers === true;

  /**
   * Handle single answer selection
   */
  const handleSingleAnswerChange = (optionIndex) => {
    if (!readOnly) {
      onChange(optionIndex);
    }
  };

  /**
   * Handle multiple answer selection
   */
  const handleMultipleAnswerChange = (optionIndex) => {
    if (readOnly) return;

    const currentAnswers = Array.isArray(answer) ? [...answer] : [];
    const indexInArray = currentAnswers.indexOf(optionIndex);

    if (indexInArray > -1) {
      // Remove if already selected
      currentAnswers.splice(indexInArray, 1);
    } else {
      // Add if not selected
      currentAnswers.push(optionIndex);
    }

    onChange(currentAnswers);
  };

  /**
   * Check if an option is selected
   */
  const isOptionSelected = (optionIndex) => {
    if (isMultipleAnswer) {
      return Array.isArray(answer) && answer.includes(optionIndex);
    }
    return answer === optionIndex;
  };

  /**
   * Check if an option is correct (for review mode)
   */
  const isOptionCorrect = (optionIndex) => {
    if (!showExplanation) return false;

    if (isMultipleAnswer) {
      return Array.isArray(question.correctAnswer) && question.correctAnswer.includes(optionIndex);
    }
    return question.correctAnswer === optionIndex;
  };

  /**
   * Get option styling
   */
  const getOptionClassName = (optionIndex) => {
    const baseClasses = "w-full p-4 border-2 rounded-lg text-left transition-all duration-200";
    const isSelected = isOptionSelected(optionIndex);
    const isCorrect = isOptionCorrect(optionIndex);

    if (showExplanation) {
      // Review mode - show correct/incorrect
      if (isCorrect) {
        return `${baseClasses} border-green-500 bg-green-50 text-green-900`;
      }
      if (isSelected && !isCorrect) {
        return `${baseClasses} border-red-500 bg-red-50 text-red-900`;
      }
      return `${baseClasses} border-gray-300 bg-gray-50 text-gray-700`;
    }

    // Normal mode
    if (isSelected) {
      return `${baseClasses} border-blue-500 bg-blue-50 text-blue-900 font-medium`;
    }

    if (readOnly) {
      return `${baseClasses} border-gray-300 bg-gray-50 text-gray-700 cursor-not-allowed`;
    }

    return `${baseClasses} border-gray-300 hover:border-gray-400 hover:bg-gray-50 cursor-pointer`;
  };

  /**
   * Get option icon
   */
  const getOptionIcon = (optionIndex) => {
    const isSelected = isOptionSelected(optionIndex);
    const isCorrect = isOptionCorrect(optionIndex);

    if (showExplanation) {
      if (isCorrect) {
        return <span className="text-green-600 text-xl">✓</span>;
      }
      if (isSelected && !isCorrect) {
        return <span className="text-red-600 text-xl">✗</span>;
      }
      return <span className="text-gray-400 text-xl">○</span>;
    }

    if (isMultipleAnswer) {
      return (
        <span className={`w-5 h-5 border-2 rounded flex items-center justify-center ${
          isSelected ? 'border-blue-500 bg-blue-500' : 'border-gray-400'
        }`}>
          {isSelected && <span className="text-white text-xs">✓</span>}
        </span>
      );
    }

    return (
      <span className={`w-5 h-5 border-2 rounded-full flex items-center justify-center ${
        isSelected ? 'border-blue-500' : 'border-gray-400'
      }`}>
        {isSelected && <span className="w-3 h-3 bg-blue-500 rounded-full"></span>}
      </span>
    );
  };

  return (
    <div className="space-y-3">
      {/* Instruction for multiple answer */}
      {isMultipleAnswer && !readOnly && (
        <p className="text-sm text-gray-600 italic mb-4">
          ℹ️ Select all that apply (multiple answers may be correct)
        </p>
      )}

      {/* Options */}
      {question.options.map((option, index) => (
        <button
          key={index}
          onClick={() => {
            if (isMultipleAnswer) {
              handleMultipleAnswerChange(index);
            } else {
              handleSingleAnswerChange(index);
            }
          }}
          disabled={readOnly}
          className={getOptionClassName(index)}
        >
          <div className="flex items-start gap-3">
            <div className="flex-shrink-0 mt-0.5">
              {getOptionIcon(index)}
            </div>
            <div className="flex-1">
              <div className="flex items-start justify-between gap-4">
                <span className="text-base">{option}</span>
                {showExplanation && isOptionCorrect(index) && (
                  <span className="flex-shrink-0 px-2 py-1 bg-green-100 text-green-800 text-xs font-semibold rounded">
                    Correct
                  </span>
                )}
              </div>
            </div>
          </div>
        </button>
      ))}

      {/* Selection summary for multiple answer */}
      {isMultipleAnswer && !readOnly && (
        <p className="text-sm text-gray-500 mt-2">
          {Array.isArray(answer) && answer.length > 0
            ? `${answer.length} option${answer.length > 1 ? 's' : ''} selected`
            : 'No options selected'}
        </p>
      )}
    </div>
  );
};

export default MultipleChoiceQuestion;
