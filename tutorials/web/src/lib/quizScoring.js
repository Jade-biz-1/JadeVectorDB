/**
 * Quiz Scoring and Grading Logic
 *
 * Provides functions for grading quiz questions, calculating scores,
 * and analyzing assessment performance.
 */

/**
 * Grade a single question
 * @param {Object} question - Question object from quiz data
 * @param {any} userAnswer - User's answer
 * @returns {Object} Grading result
 */
export function gradeQuestion(question, userAnswer) {
  const result = {
    questionId: question.id,
    type: question.type,
    points: question.points,
    earnedPoints: 0,
    isCorrect: false,
    userAnswer,
    correctAnswer: question.correctAnswer,
    explanation: question.explanation,
    partialCredit: false
  };

  if (userAnswer === null || userAnswer === undefined) {
    return result;
  }

  switch (question.type) {
    case 'multiple-choice':
      if (question.multipleAnswers) {
        result.isCorrect = gradeMultipleAnswerQuestion(question, userAnswer);
        if (result.isCorrect) {
          result.earnedPoints = question.points;
        }
      } else {
        result.isCorrect = userAnswer === question.correctAnswer;
        if (result.isCorrect) {
          result.earnedPoints = question.points;
        }
      }
      break;

    case 'true-false':
      result.isCorrect = userAnswer === question.correctAnswer;
      if (result.isCorrect) {
        result.earnedPoints = question.points;
      }
      break;

    case 'code-challenge':
      const codeResult = gradeCodeChallenge(question, userAnswer);
      result.isCorrect = codeResult.isCorrect;
      result.earnedPoints = codeResult.earnedPoints;
      result.partialCredit = codeResult.partialCredit;
      result.testResults = codeResult.testResults;
      break;

    case 'fill-blank':
      result.isCorrect = gradeFillBlank(question, userAnswer);
      if (result.isCorrect) {
        result.earnedPoints = question.points;
      }
      break;

    default:
      console.warn(`Unknown question type: ${question.type}`);
  }

  return result;
}

/**
 * Grade a multiple-answer multiple-choice question
 * @param {Object} question - Question object
 * @param {Array} userAnswer - Array of user's selected answers
 * @returns {boolean} Whether answer is correct
 */
function gradeMultipleAnswerQuestion(question, userAnswer) {
  if (!Array.isArray(userAnswer) || !Array.isArray(question.correctAnswer)) {
    return false;
  }

  // Sort both arrays for comparison
  const sortedUserAnswer = [...userAnswer].sort();
  const sortedCorrectAnswer = [...question.correctAnswer].sort();

  // Compare lengths
  if (sortedUserAnswer.length !== sortedCorrectAnswer.length) {
    return false;
  }

  // Compare each element
  return sortedUserAnswer.every((answer, index) => answer === sortedCorrectAnswer[index]);
}

/**
 * Grade a code challenge question
 * @param {Object} question - Question object
 * @param {string|Object} userAnswer - User's code or config object
 * @returns {Object} Grading result with test case results
 */
function gradeCodeChallenge(question, userAnswer) {
  const result = {
    isCorrect: false,
    earnedPoints: 0,
    partialCredit: false,
    testResults: []
  };

  if (!question.testCases || question.testCases.length === 0) {
    // Simple comparison if no test cases
    const isEqual = deepEqual(userAnswer, question.correctAnswer);
    result.isCorrect = isEqual;
    result.earnedPoints = isEqual ? question.points : 0;
    return result;
  }

  // Run test cases
  let passedTests = 0;
  for (const testCase of question.testCases) {
    const testResult = {
      description: testCase.description,
      passed: false,
      error: null
    };

    try {
      // Evaluate the validation expression
      // This is a simplified version - in production, use a safer eval alternative
      const passed = evaluateTestCase(testCase.validate, userAnswer);
      testResult.passed = passed;
      if (passed) passedTests++;
    } catch (error) {
      testResult.error = error.message;
    }

    result.testResults.push(testResult);
  }

  const totalTests = question.testCases.length;
  const allPassed = passedTests === totalTests;

  result.isCorrect = allPassed;

  // Award partial credit if some tests passed
  if (passedTests > 0) {
    const partialPoints = (question.points * passedTests) / totalTests;
    result.earnedPoints = Math.round(partialPoints);
    result.partialCredit = !allPassed && passedTests > 0;
  }

  return result;
}

/**
 * Evaluate a test case validation expression
 * @param {string} expression - Validation expression
 * @param {any} userAnswer - User's answer
 * @returns {boolean} Whether the test passed
 */
function evaluateTestCase(expression, userAnswer) {
  // Create context for evaluation
  const context = {
    config: userAnswer,
    params: userAnswer,
    vector: userAnswer,
    update: userAnswer
  };

  try {
    // Use Function constructor as a safer alternative to eval
    const func = new Function(...Object.keys(context), `return ${expression}`);
    return func(...Object.values(context));
  } catch (error) {
    console.error('Test case evaluation error:', error);
    return false;
  }
}

/**
 * Grade a fill-in-the-blank question
 * @param {Object} question - Question object
 * @param {string} userAnswer - User's answer
 * @returns {boolean} Whether answer is correct
 */
function gradeFillBlank(question, userAnswer) {
  if (typeof userAnswer !== 'string') {
    return false;
  }

  const normalizedUserAnswer = userAnswer.trim().toLowerCase();
  const normalizedCorrectAnswer = question.correctAnswer.toString().toLowerCase();

  return normalizedUserAnswer === normalizedCorrectAnswer;
}

/**
 * Deep equality comparison for objects and arrays
 * @param {any} obj1 - First object
 * @param {any} obj2 - Second object
 * @returns {boolean} Whether objects are deeply equal
 */
function deepEqual(obj1, obj2) {
  if (obj1 === obj2) return true;

  if (obj1 == null || obj2 == null) return false;
  if (typeof obj1 !== 'object' || typeof obj2 !== 'object') return obj1 === obj2;

  const keys1 = Object.keys(obj1);
  const keys2 = Object.keys(obj2);

  if (keys1.length !== keys2.length) return false;

  for (const key of keys1) {
    if (!keys2.includes(key)) return false;
    if (!deepEqual(obj1[key], obj2[key])) return false;
  }

  return true;
}

/**
 * Calculate total score from graded results
 * @param {Array} gradedResults - Array of grading results
 * @returns {Object} Score summary
 */
export function calculateTotalScore(gradedResults) {
  const totalPoints = gradedResults.reduce((sum, result) => sum + result.points, 0);
  const earnedPoints = gradedResults.reduce((sum, result) => sum + result.earnedPoints, 0);

  return {
    totalPoints,
    earnedPoints,
    percentage: totalPoints > 0 ? Math.round((earnedPoints / totalPoints) * 100) : 0,
    correctCount: gradedResults.filter(r => r.isCorrect).length,
    totalQuestions: gradedResults.length
  };
}

/**
 * Determine if the assessment is passing
 * @param {number} score - Score percentage (0-100)
 * @param {number} minScore - Minimum passing score (default: 70)
 * @returns {boolean} Whether the score is passing
 */
export function isPassing(score, minScore = 70) {
  return score >= minScore;
}

/**
 * Generate performance analysis
 * @param {Array} gradedResults - Array of grading results
 * @returns {Object} Performance analysis
 */
export function analyzePerformance(gradedResults) {
  const analysis = {
    byDifficulty: {
      easy: { total: 0, correct: 0, percentage: 0 },
      medium: { total: 0, correct: 0, percentage: 0 },
      hard: { total: 0, correct: 0, percentage: 0 }
    },
    byType: {},
    strengths: [],
    weaknesses: [],
    recommendations: []
  };

  // Analyze by difficulty and type
  gradedResults.forEach(result => {
    const question = result;

    // By difficulty
    if (question.difficulty) {
      const difficulty = question.difficulty.toLowerCase();
      if (analysis.byDifficulty[difficulty]) {
        analysis.byDifficulty[difficulty].total++;
        if (result.isCorrect) {
          analysis.byDifficulty[difficulty].correct++;
        }
      }
    }

    // By type
    if (!analysis.byType[result.type]) {
      analysis.byType[result.type] = { total: 0, correct: 0, percentage: 0 };
    }
    analysis.byType[result.type].total++;
    if (result.isCorrect) {
      analysis.byType[result.type].correct++;
    }
  });

  // Calculate percentages
  Object.keys(analysis.byDifficulty).forEach(difficulty => {
    const data = analysis.byDifficulty[difficulty];
    if (data.total > 0) {
      data.percentage = Math.round((data.correct / data.total) * 100);
    }
  });

  Object.keys(analysis.byType).forEach(type => {
    const data = analysis.byType[type];
    if (data.total > 0) {
      data.percentage = Math.round((data.correct / data.total) * 100);
    }
  });

  // Identify strengths and weaknesses
  Object.entries(analysis.byDifficulty).forEach(([difficulty, data]) => {
    if (data.total > 0) {
      if (data.percentage >= 80) {
        analysis.strengths.push(`Strong performance on ${difficulty} questions (${data.percentage}%)`);
      } else if (data.percentage < 60) {
        analysis.weaknesses.push(`Needs improvement on ${difficulty} questions (${data.percentage}%)`);
      }
    }
  });

  // Generate recommendations
  if (analysis.byDifficulty.easy.percentage < 80) {
    analysis.recommendations.push('Review fundamental concepts and basics');
  }
  if (analysis.byDifficulty.medium.percentage < 70) {
    analysis.recommendations.push('Practice intermediate topics and hands-on examples');
  }
  if (analysis.byDifficulty.hard.percentage < 60) {
    analysis.recommendations.push('Study advanced features and best practices');
  }

  // Type-specific recommendations
  Object.entries(analysis.byType).forEach(([type, data]) => {
    if (data.percentage < 60) {
      if (type === 'code-challenge') {
        analysis.recommendations.push('Practice more code examples and API usage');
      } else if (type === 'multiple-choice') {
        analysis.recommendations.push('Review conceptual understanding of key topics');
      }
    }
  });

  return analysis;
}

/**
 * Calculate time metrics
 * @param {number} startTime - Start timestamp
 * @param {number} endTime - End timestamp
 * @returns {Object} Time metrics
 */
export function calculateTimeMetrics(startTime, endTime) {
  const totalMs = endTime - startTime;
  const totalSeconds = Math.floor(totalMs / 1000);
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;

  return {
    totalMs,
    totalSeconds,
    minutes,
    seconds,
    formatted: `${minutes}:${seconds.toString().padStart(2, '0')}`
  };
}

/**
 * Generate a grade letter based on percentage
 * @param {number} percentage - Score percentage (0-100)
 * @returns {string} Grade letter
 */
export function getGradeLetter(percentage) {
  if (percentage >= 90) return 'A';
  if (percentage >= 80) return 'B';
  if (percentage >= 70) return 'C';
  if (percentage >= 60) return 'D';
  return 'F';
}

/**
 * Get performance level description
 * @param {number} percentage - Score percentage (0-100)
 * @returns {Object} Performance level info
 */
export function getPerformanceLevel(percentage) {
  if (percentage >= 90) {
    return {
      level: 'Excellent',
      description: 'Outstanding mastery of the material',
      color: '#22c55e' // green
    };
  }
  if (percentage >= 80) {
    return {
      level: 'Very Good',
      description: 'Strong understanding with minor gaps',
      color: '#3b82f6' // blue
    };
  }
  if (percentage >= 70) {
    return {
      level: 'Good',
      description: 'Adequate understanding to proceed',
      color: '#eab308' // yellow
    };
  }
  if (percentage >= 60) {
    return {
      level: 'Fair',
      description: 'Basic understanding, review recommended',
      color: '#f97316' // orange
    };
  }
  return {
    level: 'Needs Improvement',
    description: 'Additional study required',
    color: '#ef4444' // red
  };
}

/**
 * Compare two assessment attempts
 * @param {Object} attempt1 - First attempt
 * @param {Object} attempt2 - Second attempt
 * @returns {Object} Comparison result
 */
export function compareAttempts(attempt1, attempt2) {
  return {
    scoreDifference: attempt2.score - attempt1.score,
    scoreImprovement: attempt2.score > attempt1.score,
    timeChange: attempt2.timeSpent - attempt1.timeSpent,
    fasterCompletion: attempt2.timeSpent < attempt1.timeSpent,
    improvementPercentage: Math.round(
      ((attempt2.score - attempt1.score) / attempt1.score) * 100
    )
  };
}

export default {
  gradeQuestion,
  calculateTotalScore,
  isPassing,
  analyzePerformance,
  calculateTimeMetrics,
  getGradeLetter,
  getPerformanceLevel,
  compareAttempts
};
