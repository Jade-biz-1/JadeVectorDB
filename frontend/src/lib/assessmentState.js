/**
 * Assessment State Management
 *
 * Manages quiz and assessment state for the JadeVectorDB tutorial system.
 * Handles assessment sessions, results, and persistence.
 */

class AssessmentStateManager {
  constructor() {
    this.storageKey = 'jadevectordb_assessments';
    this.currentSession = null;
  }

  /**
   * Initialize a new assessment session for a module
   * @param {string} moduleId - Module identifier (e.g., 'module1')
   * @param {Object} quizData - Quiz data including questions
   * @returns {Object} Assessment session
   */
  initAssessment(moduleId, quizData) {
    this.currentSession = {
      moduleId,
      moduleName: quizData.moduleName,
      startTime: Date.now(),
      endTime: null,
      answers: {},
      currentQuestionIndex: 0,
      totalQuestions: quizData.questions.length,
      isComplete: false,
      score: null,
      passed: null
    };

    return this.currentSession;
  }

  /**
   * Get the current assessment session
   * @returns {Object|null} Current session or null
   */
  getCurrentAssessment() {
    return this.currentSession;
  }

  /**
   * Save answer for a specific question
   * @param {string} questionId - Question identifier
   * @param {any} answer - User's answer
   */
  saveAnswer(questionId, answer) {
    if (!this.currentSession) {
      throw new Error('No active assessment session');
    }

    this.currentSession.answers[questionId] = {
      answer,
      timestamp: Date.now()
    };
  }

  /**
   * Get answer for a specific question
   * @param {string} questionId - Question identifier
   * @returns {any|null} User's answer or null
   */
  getAnswer(questionId) {
    if (!this.currentSession || !this.currentSession.answers[questionId]) {
      return null;
    }
    return this.currentSession.answers[questionId].answer;
  }

  /**
   * Move to next question
   * @returns {number} New question index
   */
  nextQuestion() {
    if (!this.currentSession) {
      throw new Error('No active assessment session');
    }

    this.currentSession.currentQuestionIndex++;
    return this.currentSession.currentQuestionIndex;
  }

  /**
   * Move to previous question
   * @returns {number} New question index
   */
  previousQuestion() {
    if (!this.currentSession) {
      throw new Error('No active assessment session');
    }

    if (this.currentSession.currentQuestionIndex > 0) {
      this.currentSession.currentQuestionIndex--;
    }
    return this.currentSession.currentQuestionIndex;
  }

  /**
   * Go to specific question
   * @param {number} index - Question index
   * @returns {number} New question index
   */
  goToQuestion(index) {
    if (!this.currentSession) {
      throw new Error('No active assessment session');
    }

    if (index >= 0 && index < this.currentSession.totalQuestions) {
      this.currentSession.currentQuestionIndex = index;
    }
    return this.currentSession.currentQuestionIndex;
  }

  /**
   * Complete the assessment session
   * @param {Object} result - Assessment result including score
   */
  completeAssessment(result) {
    if (!this.currentSession) {
      throw new Error('No active assessment session');
    }

    this.currentSession.endTime = Date.now();
    this.currentSession.isComplete = true;
    this.currentSession.score = result.score;
    this.currentSession.passed = result.passed;
    this.currentSession.result = result;

    // Save to persistent storage
    this.saveAssessmentResult(this.currentSession);

    return this.currentSession;
  }

  /**
   * Save assessment result to persistent storage
   * @param {Object} session - Completed assessment session
   * @returns {boolean} Success status
   */
  saveAssessmentResult(session) {
    try {
      const history = this.getAssessmentHistory();

      // Find existing results for this module
      const moduleHistory = history[session.moduleId] || [];

      // Add new result
      moduleHistory.push({
        attemptNumber: moduleHistory.length + 1,
        date: session.endTime,
        score: session.score,
        passed: session.passed,
        timeSpent: session.endTime - session.startTime,
        totalQuestions: session.totalQuestions,
        answers: session.answers
      });

      history[session.moduleId] = moduleHistory;

      // Save to localStorage
      localStorage.setItem(this.storageKey, JSON.stringify(history));

      // Also update the main tutorial state
      this.updateTutorialState(session.moduleId, session.score, session.passed);

      return true;
    } catch (error) {
      console.error('Failed to save assessment result:', error);
      return false;
    }
  }

  /**
   * Update tutorial state with assessment results
   * @param {string} moduleId - Module identifier
   * @param {number} score - Assessment score
   * @param {boolean} passed - Whether the assessment was passed
   */
  updateTutorialState(moduleId, score, passed) {
    try {
      const tutorialState = JSON.parse(
        localStorage.getItem('jadevectordb_tutorial_state') || '{}'
      );

      if (!tutorialState.assessments) {
        tutorialState.assessments = {};
      }

      // Get module number from moduleId (e.g., 'module1' -> 0)
      const moduleNumber = parseInt(moduleId.replace('module', '')) - 1;

      if (!tutorialState.assessments[moduleNumber]) {
        tutorialState.assessments[moduleNumber] = {
          attempts: 0,
          bestScore: 0,
          lastAttempt: null,
          passed: false
        };
      }

      const moduleAssessment = tutorialState.assessments[moduleNumber];
      moduleAssessment.attempts++;
      moduleAssessment.lastAttempt = Date.now();

      if (score > moduleAssessment.bestScore) {
        moduleAssessment.bestScore = score;
      }

      if (passed) {
        moduleAssessment.passed = true;

        // Unlock next module if this one is passed
        if (tutorialState.modules && tutorialState.modules[moduleNumber + 1]) {
          tutorialState.modules[moduleNumber + 1].unlocked = true;
        }
      }

      localStorage.setItem('jadevectordb_tutorial_state', JSON.stringify(tutorialState));
    } catch (error) {
      console.error('Failed to update tutorial state:', error);
    }
  }

  /**
   * Get assessment history for a specific module
   * @param {string} moduleId - Module identifier
   * @returns {Array} Array of assessment attempts
   */
  getModuleHistory(moduleId) {
    const history = this.getAssessmentHistory();
    return history[moduleId] || [];
  }

  /**
   * Get complete assessment history
   * @returns {Object} Object containing all assessment history
   */
  getAssessmentHistory() {
    try {
      const data = localStorage.getItem(this.storageKey);
      return data ? JSON.parse(data) : {};
    } catch (error) {
      console.error('Failed to load assessment history:', error);
      return {};
    }
  }

  /**
   * Get the best score for a module
   * @param {string} moduleId - Module identifier
   * @returns {number|null} Best score or null if no attempts
   */
  getBestScore(moduleId) {
    const moduleHistory = this.getModuleHistory(moduleId);
    if (moduleHistory.length === 0) {
      return null;
    }

    return Math.max(...moduleHistory.map(attempt => attempt.score));
  }

  /**
   * Get the last attempt for a module
   * @param {string} moduleId - Module identifier
   * @returns {Object|null} Last attempt data or null
   */
  getLastAttempt(moduleId) {
    const moduleHistory = this.getModuleHistory(moduleId);
    if (moduleHistory.length === 0) {
      return null;
    }

    return moduleHistory[moduleHistory.length - 1];
  }

  /**
   * Check if a module assessment has been passed
   * @param {string} moduleId - Module identifier
   * @returns {boolean} Whether the module has been passed
   */
  hasPassedModule(moduleId) {
    const moduleHistory = this.getModuleHistory(moduleId);
    return moduleHistory.some(attempt => attempt.passed);
  }

  /**
   * Get overall progress across all assessments
   * @returns {Object} Progress summary
   */
  getOverallProgress() {
    const history = this.getAssessmentHistory();
    const modules = ['module1', 'module2', 'module3', 'module4', 'module5', 'module6'];

    const progress = {
      totalModules: modules.length,
      completedModules: 0,
      passedModules: 0,
      totalAttempts: 0,
      averageScore: 0,
      bestScores: {}
    };

    let totalScore = 0;
    let scoredModules = 0;

    modules.forEach(moduleId => {
      const moduleHistory = history[moduleId] || [];

      if (moduleHistory.length > 0) {
        progress.completedModules++;
        progress.totalAttempts += moduleHistory.length;

        const bestScore = Math.max(...moduleHistory.map(a => a.score));
        progress.bestScores[moduleId] = bestScore;
        totalScore += bestScore;
        scoredModules++;

        if (moduleHistory.some(a => a.passed)) {
          progress.passedModules++;
        }
      }
    });

    if (scoredModules > 0) {
      progress.averageScore = Math.round(totalScore / scoredModules);
    }

    return progress;
  }

  /**
   * Retry an assessment (clear current session, keep history)
   * @param {string} moduleId - Module identifier
   */
  retryAssessment(moduleId) {
    this.currentSession = null;
    return { moduleId, retry: true };
  }

  /**
   * Clear all assessment history
   * @returns {boolean} Success status
   */
  clearHistory() {
    try {
      localStorage.removeItem(this.storageKey);
      this.currentSession = null;
      return true;
    } catch (error) {
      console.error('Failed to clear assessment history:', error);
      return false;
    }
  }

  /**
   * Export assessment data
   * @returns {string} JSON string of assessment data
   */
  exportData() {
    const history = this.getAssessmentHistory();
    return JSON.stringify(history, null, 2);
  }

  /**
   * Import assessment data
   * @param {string} jsonData - JSON string of assessment data
   * @returns {boolean} Success status
   */
  importData(jsonData) {
    try {
      const data = JSON.parse(jsonData);
      localStorage.setItem(this.storageKey, JSON.stringify(data));
      return true;
    } catch (error) {
      console.error('Failed to import assessment data:', error);
      return false;
    }
  }

  /**
   * Get statistics for a specific module
   * @param {string} moduleId - Module identifier
   * @returns {Object} Module statistics
   */
  getModuleStatistics(moduleId) {
    const moduleHistory = this.getModuleHistory(moduleId);

    if (moduleHistory.length === 0) {
      return {
        attempts: 0,
        passed: false,
        bestScore: null,
        averageScore: null,
        totalTimeSpent: 0,
        lastAttempt: null
      };
    }

    const scores = moduleHistory.map(a => a.score);
    const totalTime = moduleHistory.reduce((sum, a) => sum + (a.timeSpent || 0), 0);

    return {
      attempts: moduleHistory.length,
      passed: moduleHistory.some(a => a.passed),
      bestScore: Math.max(...scores),
      averageScore: Math.round(scores.reduce((sum, s) => sum + s, 0) / scores.length),
      totalTimeSpent: totalTime,
      lastAttempt: moduleHistory[moduleHistory.length - 1].date,
      improvement: moduleHistory.length > 1 ?
        scores[scores.length - 1] - scores[0] : 0
    };
  }
}

// Export singleton instance
const assessmentState = new AssessmentStateManager();
export default assessmentState;
