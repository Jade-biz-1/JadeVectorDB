/**
 * Assessment Engine - Core logic for quiz system
 * Handles scoring, validation, progress tracking, and persistence
 */

export class AssessmentEngine {
  constructor() {
    this.STORAGE_KEY = 'jadevectordb_quiz_progress';
    this.RESULTS_KEY = 'jadevectordb_quiz_results';
  }

  /**
   * Validate answer for different question types
   */
  validateAnswer(question, userAnswer) {
    switch (question.type) {
      case 'multiple-choice':
        return userAnswer === question.correctAnswer;

      case 'code-completion':
        return this.validateCodeCompletion(question.correctAnswer, userAnswer);

      case 'debugging':
        return this.validateDebugging(question.correctAnswer, userAnswer);

      case 'scenario-based':
        return userAnswer === question.correctAnswer;

      default:
        return false;
    }
  }

  /**
   * Validate code completion (flexible matching)
   */
  validateCodeCompletion(correct, userAnswer) {
    if (!userAnswer || typeof userAnswer !== 'string') return false;

    // Normalize whitespace and compare
    const normalizeCode = (code) => code
      .replace(/\s+/g, ' ')
      .replace(/\s*([(){}\[\],;:])\s*/g, '$1')
      .trim()
      .toLowerCase();

    const normalizedCorrect = normalizeCode(correct);
    const normalizedUser = normalizeCode(userAnswer);

    // Check for exact match or key elements
    if (normalizedCorrect === normalizedUser) return true;

    // Check if all key elements are present
    const keyElements = this.extractKeyElements(correct);
    return keyElements.every(element => normalizedUser.includes(element.toLowerCase()));
  }

  /**
   * Extract key elements from code for validation
   */
  extractKeyElements(code) {
    // Extract function calls, method names, important keywords
    const elements = [];

    // Match function calls: functionName(
    const functionCalls = code.match(/\w+\s*\(/g);
    if (functionCalls) {
      elements.push(...functionCalls.map(f => f.replace(/\s*\(/, '')));
    }

    // Match method calls: .methodName(
    const methodCalls = code.match(/\.\w+\s*\(/g);
    if (methodCalls) {
      elements.push(...methodCalls.map(m => m.replace(/\.\s*/, '').replace(/\s*\(/, '')));
    }

    return elements;
  }

  /**
   * Validate debugging answer (checks for specific fixes)
   */
  validateDebugging(correct, userAnswer) {
    if (!userAnswer || typeof userAnswer !== 'string') return false;

    // Check if the corrected code contains the fix
    return userAnswer.includes(correct) ||
           this.validateCodeCompletion(correct, userAnswer);
  }

  /**
   * Calculate score for quiz results
   */
  calculateScore(questions, userAnswers) {
    let totalPoints = 0;
    let earnedPoints = 0;
    const results = [];

    questions.forEach((question, index) => {
      totalPoints += question.points;
      const userAnswer = userAnswers[index];
      const isCorrect = this.validateAnswer(question, userAnswer);

      if (isCorrect) {
        earnedPoints += question.points;
      }

      results.push({
        questionId: question.id,
        question: question.question,
        userAnswer,
        correctAnswer: question.correctAnswer,
        isCorrect,
        points: isCorrect ? question.points : 0,
        maxPoints: question.points,
        explanation: question.explanation
      });
    });

    const percentage = totalPoints > 0 ? (earnedPoints / totalPoints) * 100 : 0;

    return {
      totalPoints,
      earnedPoints,
      percentage: Math.round(percentage * 10) / 10, // Round to 1 decimal
      results,
      passed: percentage >= 70 // Default passing score
    };
  }

  /**
   * Save quiz progress to localStorage
   */
  saveProgress(moduleId, currentQuestionIndex, userAnswers, startTime) {
    const progress = {
      moduleId,
      currentQuestionIndex,
      userAnswers,
      startTime,
      lastUpdated: Date.now()
    };

    try {
      if (typeof window === 'undefined' || typeof localStorage === 'undefined') {
        return false;
      }
      localStorage.setItem(this.STORAGE_KEY, JSON.stringify(progress));
      return true;
    } catch (error) {
      console.error('Failed to save quiz progress:', error);
      return false;
    }
  }

  /**
   * Load quiz progress from localStorage
   */
  loadProgress() {
    try {
      if (typeof window === 'undefined' || typeof localStorage === 'undefined') {
        return null;
      }
      const progress = localStorage.getItem(this.STORAGE_KEY);
      return progress ? JSON.parse(progress) : null;
    } catch (error) {
      console.error('Failed to load quiz progress:', error);
      return null;
    }
  }

  /**
   * Clear quiz progress
   */
  clearProgress() {
    try {
      if (typeof window === 'undefined' || typeof localStorage === 'undefined') {
        return false;
      }
      localStorage.removeItem(this.STORAGE_KEY);
      return true;
    } catch (error) {
      console.error('Failed to clear quiz progress:', error);
      return false;
    }
  }

  /**
   * Save quiz results
   */
  saveResults(moduleId, scoreData, timeSpent) {
    try {
      if (typeof window === 'undefined' || typeof localStorage === 'undefined') {
        return false;
      }
      const results = this.loadAllResults();

      const result = {
        moduleId,
        ...scoreData,
        timeSpent,
        completedAt: Date.now(),
        timestamp: new Date().toISOString()
      };

      // Add to results array
      if (!results[moduleId]) {
        results[moduleId] = [];
      }
      results[moduleId].push(result);

      localStorage.setItem(this.RESULTS_KEY, JSON.stringify(results));
      return true;
    } catch (error) {
      console.error('Failed to save quiz results:', error);
      return false;
    }
  }

  /**
   * Load all quiz results
   */
  loadAllResults() {
    try {
      // Check if running in browser environment
      if (typeof window === 'undefined' || typeof localStorage === 'undefined') {
        return {};
      }
      const results = localStorage.getItem(this.RESULTS_KEY);
      return results ? JSON.parse(results) : {};
    } catch (error) {
      console.error('Failed to load quiz results:', error);
      return {};
    }
  }

  /**
   * Load results for specific module
   */
  loadModuleResults(moduleId) {
    const allResults = this.loadAllResults();
    return allResults[moduleId] || [];
  }

  /**
   * Get best score for a module
   */
  getBestScore(moduleId) {
    const results = this.loadModuleResults(moduleId);
    if (results.length === 0) return null;

    return results.reduce((best, current) => {
      return current.percentage > best.percentage ? current : best;
    });
  }

  /**
   * Get quiz statistics
   */
  getStatistics(moduleId = null) {
    const allResults = this.loadAllResults();

    if (moduleId) {
      const moduleResults = allResults[moduleId] || [];
      return this.calculateModuleStats(moduleResults);
    }

    // Overall statistics
    const stats = {
      totalQuizzes: 0,
      averageScore: 0,
      passedQuizzes: 0,
      totalTimeSpent: 0,
      moduleStats: {}
    };

    Object.entries(allResults).forEach(([module, results]) => {
      const moduleStats = this.calculateModuleStats(results);
      stats.moduleStats[module] = moduleStats;
      stats.totalQuizzes += results.length;
      stats.passedQuizzes += results.filter(r => r.passed).length;
      stats.totalTimeSpent += results.reduce((sum, r) => sum + (r.timeSpent || 0), 0);
    });

    if (stats.totalQuizzes > 0) {
      const allScores = Object.values(allResults)
        .flat()
        .map(r => r.percentage);
      stats.averageScore = allScores.reduce((sum, score) => sum + score, 0) / allScores.length;
    }

    return stats;
  }

  /**
   * Calculate statistics for a specific module
   */
  calculateModuleStats(results) {
    if (results.length === 0) {
      return {
        attempts: 0,
        bestScore: 0,
        averageScore: 0,
        passed: false,
        totalTimeSpent: 0
      };
    }

    const scores = results.map(r => r.percentage);
    const bestScore = Math.max(...scores);
    const averageScore = scores.reduce((sum, score) => sum + score, 0) / scores.length;
    const totalTimeSpent = results.reduce((sum, r) => sum + (r.timeSpent || 0), 0);

    return {
      attempts: results.length,
      bestScore: Math.round(bestScore * 10) / 10,
      averageScore: Math.round(averageScore * 10) / 10,
      passed: bestScore >= 70,
      totalTimeSpent,
      lastAttempt: results[results.length - 1]
    };
  }

  /**
   * Format time in seconds to human-readable format
   */
  formatTime(seconds) {
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = seconds % 60;

    if (minutes === 0) {
      return `${remainingSeconds}s`;
    }

    return `${minutes}m ${remainingSeconds}s`;
  }

  /**
   * Check if time limit exceeded
   */
  isTimeExpired(startTime, timeLimit) {
    const elapsed = Math.floor((Date.now() - startTime) / 1000);
    return elapsed >= timeLimit;
  }

  /**
   * Get remaining time
   */
  getRemainingTime(startTime, timeLimit) {
    const elapsed = Math.floor((Date.now() - startTime) / 1000);
    return Math.max(0, timeLimit - elapsed);
  }

  /**
   * Get difficulty-based score multiplier
   */
  getDifficultyMultiplier(difficulty) {
    const multipliers = {
      easy: 1.0,
      medium: 1.5,
      hard: 2.0
    };
    return multipliers[difficulty] || 1.0;
  }

  /**
   * Generate performance feedback
   */
  generateFeedback(percentage, moduleId) {
    const feedback = {
      score: percentage,
      level: '',
      message: '',
      suggestions: []
    };

    if (percentage >= 90) {
      feedback.level = 'excellent';
      feedback.message = '🎉 Outstanding! You have mastered this module!';
      feedback.suggestions = [
        'Consider helping others in the community',
        'Explore advanced features and edge cases',
        'Try building a real-world project with these concepts'
      ];
    } else if (percentage >= 70) {
      feedback.level = 'good';
      feedback.message = '✅ Great job! You passed the assessment!';
      feedback.suggestions = [
        'Review questions you got wrong',
        'Practice the concepts you struggled with',
        'Try the quiz again to improve your score'
      ];
    } else if (percentage >= 50) {
      feedback.level = 'needs-improvement';
      feedback.message = '📚 You\'re getting there! Review the material and try again.';
      feedback.suggestions = [
        'Re-read the tutorial content for this module',
        'Practice with the interactive examples',
        'Focus on questions you got wrong',
        'Take the quiz again after reviewing'
      ];
    } else {
      feedback.level = 'needs-review';
      feedback.message = '🔄 Let\'s review the fundamentals together.';
      feedback.suggestions = [
        'Go back and complete the tutorial exercises',
        'Read the documentation carefully',
        'Practice with simpler examples first',
        'Ask for help in the community if needed'
      ];
    }

    return feedback;
  }

  /**
   * Export results as JSON
   */
  exportResults(moduleId = null) {
    const allResults = this.loadAllResults();
    const data = moduleId ? { [moduleId]: allResults[moduleId] } : allResults;

    return {
      exportDate: new Date().toISOString(),
      version: '1.0',
      results: data,
      statistics: this.getStatistics(moduleId)
    };
  }

  /**
   * Clear all results
   */
  clearAllResults() {
    try {
      if (typeof window === 'undefined' || typeof localStorage === 'undefined') {
        return false;
      }
      localStorage.removeItem(this.RESULTS_KEY);
      return true;
    } catch (error) {
      console.error('Failed to clear results:', error);
      return false;
    }
  }
}

// Create singleton instance
const assessmentEngine = new AssessmentEngine();
export default assessmentEngine;
