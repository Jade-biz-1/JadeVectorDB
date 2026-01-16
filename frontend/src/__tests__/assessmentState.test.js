/**
 * Unit tests for assessmentState.js
 * Updated to match the actual AssessmentStateManager API
 */

import assessmentState from '../lib/assessmentState';

// Mock localStorage
const localStorageMock = (() => {
  let store = {};
  return {
    getItem: (key) => store[key] || null,
    setItem: (key, value) => { store[key] = value.toString(); },
    removeItem: (key) => { delete store[key]; },
    clear: () => { store = {}; }
  };
})();

global.localStorage = localStorageMock;

describe('assessmentState', () => {
  beforeEach(() => {
    localStorage.clear();
    // Reset the current session
    assessmentState.currentSession = null;
  });

  describe('initAssessment', () => {
    it('should initialize a new assessment session', () => {
      const quizData = {
        moduleId: 'module1',
        moduleName: 'Test Module',
        questions: [
          { id: 'q1', type: 'multiple-choice', question: 'Test?', points: 10 }
        ]
      };

      const session = assessmentState.initAssessment('module1', quizData);

      expect(session.moduleId).toBe('module1');
      expect(session.moduleName).toBe('Test Module');
      expect(session.answers).toEqual({});
      expect(session.startTime).toBeDefined();
      expect(session.isComplete).toBe(false);
      expect(session.totalQuestions).toBe(1);
    });

    it('should reset previous session when starting new one', () => {
      const quizData = { moduleName: 'Test', questions: [{ id: 'q1' }] };
      const session1 = assessmentState.initAssessment('module1', quizData);
      const session2 = assessmentState.initAssessment('module2', quizData);

      expect(session2.moduleId).toBe('module2');
      expect(assessmentState.getCurrentAssessment().moduleId).toBe('module2');
    });
  });

  describe('saveAnswer', () => {
    it('should save an answer to current session', () => {
      const quizData = { moduleName: 'Test', questions: [{ id: 'q1' }] };
      assessmentState.initAssessment('module1', quizData);

      assessmentState.saveAnswer('q1', 0);
      const session = assessmentState.getCurrentAssessment();

      expect(session.answers.q1.answer).toBe(0);
      expect(session.answers.q1.timestamp).toBeDefined();
    });

    it('should update existing answer', () => {
      const quizData = { moduleName: 'Test', questions: [{ id: 'q1' }] };
      assessmentState.initAssessment('module1', quizData);

      assessmentState.saveAnswer('q1', 0);
      assessmentState.saveAnswer('q1', 2);
      const session = assessmentState.getCurrentAssessment();

      expect(session.answers.q1.answer).toBe(2);
    });

    it('should throw error when no active session', () => {
      expect(() => assessmentState.saveAnswer('q1', 0)).toThrow('No active assessment session');
    });
  });

  describe('getAnswer', () => {
    it('should return answer for a question', () => {
      const quizData = { moduleName: 'Test', questions: [{ id: 'q1' }] };
      assessmentState.initAssessment('module1', quizData);
      assessmentState.saveAnswer('q1', 2);

      const answer = assessmentState.getAnswer('q1');
      expect(answer).toBe(2);
    });

    it('should return null for unanswered question', () => {
      const quizData = { moduleName: 'Test', questions: [{ id: 'q1' }] };
      assessmentState.initAssessment('module1', quizData);

      const answer = assessmentState.getAnswer('q1');
      expect(answer).toBeNull();
    });
  });

  describe('completeAssessment', () => {
    it('should mark assessment as complete', () => {
      const quizData = { moduleName: 'Test', questions: [{ id: 'q1' }] };
      assessmentState.initAssessment('module1', quizData);

      const result = {
        score: 85,
        passed: true,
        totalPoints: 100,
        earnedPoints: 85
      };

      const completedSession = assessmentState.completeAssessment(result);

      expect(completedSession.isComplete).toBe(true);
      expect(completedSession.result).toEqual(result);
      expect(completedSession.endTime).toBeDefined();
      expect(completedSession.score).toBe(85);
      expect(completedSession.passed).toBe(true);
    });

    it('should add to assessment history', () => {
      const quizData = { moduleName: 'Test', questions: [{ id: 'q1' }] };
      assessmentState.initAssessment('module1', quizData);

      const result = { score: 85, passed: true };
      assessmentState.completeAssessment(result);

      const history = assessmentState.getModuleHistory('module1');
      expect(history.length).toBe(1);
      expect(history[0].score).toBe(85);
    });

    it('should throw error when no active session', () => {
      expect(() => assessmentState.completeAssessment({ score: 80 })).toThrow('No active assessment session');
    });
  });

  describe('getModuleHistory', () => {
    it('should return empty array for module with no history', () => {
      const history = assessmentState.getModuleHistory('module99');
      expect(history).toEqual([]);
    });

    it('should return all attempts for a module', () => {
      const quizData = { moduleName: 'Test', questions: [{ id: 'q1' }] };

      // First attempt
      assessmentState.initAssessment('module1', quizData);
      assessmentState.completeAssessment({ score: 60, passed: false });

      // Second attempt
      assessmentState.initAssessment('module1', quizData);
      assessmentState.completeAssessment({ score: 80, passed: true });

      const history = assessmentState.getModuleHistory('module1');
      expect(history.length).toBe(2);
      expect(history[0].score).toBe(60);
      expect(history[1].score).toBe(80);
    });
  });

  describe('getBestScore', () => {
    it('should return null for module with no attempts', () => {
      const bestScore = assessmentState.getBestScore('module99');
      expect(bestScore).toBeNull();
    });

    it('should return highest score from multiple attempts', () => {
      const quizData = { moduleName: 'Test', questions: [{ id: 'q1' }] };

      assessmentState.initAssessment('module1', quizData);
      assessmentState.completeAssessment({ score: 60, passed: false });

      assessmentState.initAssessment('module1', quizData);
      assessmentState.completeAssessment({ score: 85, passed: true });

      assessmentState.initAssessment('module1', quizData);
      assessmentState.completeAssessment({ score: 75, passed: true });

      const bestScore = assessmentState.getBestScore('module1');
      expect(bestScore).toBe(85);
    });
  });

  describe('hasPassedModule', () => {
    it('should return false if module never attempted', () => {
      expect(assessmentState.hasPassedModule('module99')).toBe(false);
    });

    it('should return false if all attempts failed', () => {
      const quizData = { moduleName: 'Test', questions: [{ id: 'q1' }] };

      assessmentState.initAssessment('module1', quizData);
      assessmentState.completeAssessment({ score: 60, passed: false });

      expect(assessmentState.hasPassedModule('module1')).toBe(false);
    });

    it('should return true if at least one attempt passed', () => {
      const quizData = { moduleName: 'Test', questions: [{ id: 'q1' }] };

      assessmentState.initAssessment('module1', quizData);
      assessmentState.completeAssessment({ score: 60, passed: false });

      assessmentState.initAssessment('module1', quizData);
      assessmentState.completeAssessment({ score: 80, passed: true });

      expect(assessmentState.hasPassedModule('module1')).toBe(true);
    });
  });

  describe('getOverallProgress', () => {
    it('should calculate correct progress stats', () => {
      const quizData1 = { moduleName: 'Module 1', questions: [{ id: 'q1' }] };
      const quizData2 = { moduleName: 'Module 2', questions: [{ id: 'q1' }] };

      // Complete module1
      assessmentState.initAssessment('module1', quizData1);
      assessmentState.completeAssessment({ score: 80, passed: true });

      // Complete module2
      assessmentState.initAssessment('module2', quizData2);
      assessmentState.completeAssessment({ score: 90, passed: true });

      const progress = assessmentState.getOverallProgress();

      expect(progress.totalModules).toBe(6); // Fixed set of modules
      expect(progress.completedModules).toBe(2);
      expect(progress.passedModules).toBe(2);
      expect(progress.bestScores.module1).toBe(80);
      expect(progress.bestScores.module2).toBe(90);
      expect(progress.averageScore).toBe(85);
    });

    it('should return zero stats when no assessments completed', () => {
      const progress = assessmentState.getOverallProgress();

      expect(progress.completedModules).toBe(0);
      expect(progress.averageScore).toBe(0);
    });
  });

  describe('clearHistory', () => {
    it('should clear all assessment history', () => {
      const quizData = { moduleName: 'Test', questions: [{ id: 'q1' }] };
      assessmentState.initAssessment('module1', quizData);
      assessmentState.completeAssessment({ score: 80, passed: true });

      assessmentState.clearHistory();

      const history = assessmentState.getModuleHistory('module1');
      expect(history).toEqual([]);
    });

    it('should also clear current session', () => {
      const quizData = { moduleName: 'Test', questions: [{ id: 'q1' }] };
      assessmentState.initAssessment('module1', quizData);

      assessmentState.clearHistory();

      expect(assessmentState.getCurrentAssessment()).toBeNull();
    });
  });

  describe('getModuleStatistics', () => {
    it('should return empty stats for module with no attempts', () => {
      const stats = assessmentState.getModuleStatistics('module99');

      expect(stats.attempts).toBe(0);
      expect(stats.passed).toBe(false);
      expect(stats.bestScore).toBeNull();
    });

    it('should calculate correct statistics for module', () => {
      const quizData = { moduleName: 'Test', questions: [{ id: 'q1' }] };

      assessmentState.initAssessment('module1', quizData);
      assessmentState.completeAssessment({ score: 60, passed: false });

      assessmentState.initAssessment('module1', quizData);
      assessmentState.completeAssessment({ score: 80, passed: true });

      const stats = assessmentState.getModuleStatistics('module1');

      expect(stats.attempts).toBe(2);
      expect(stats.passed).toBe(true);
      expect(stats.bestScore).toBe(80);
      expect(stats.averageScore).toBe(70);
    });
  });

  describe('navigation', () => {
    it('should move to next question', () => {
      const quizData = { moduleName: 'Test', questions: [{ id: 'q1' }, { id: 'q2' }] };
      assessmentState.initAssessment('module1', quizData);

      const newIndex = assessmentState.nextQuestion();
      expect(newIndex).toBe(1);
    });

    it('should move to previous question', () => {
      const quizData = { moduleName: 'Test', questions: [{ id: 'q1' }, { id: 'q2' }] };
      assessmentState.initAssessment('module1', quizData);
      assessmentState.nextQuestion();

      const newIndex = assessmentState.previousQuestion();
      expect(newIndex).toBe(0);
    });

    it('should go to specific question', () => {
      const quizData = { moduleName: 'Test', questions: [{ id: 'q1' }, { id: 'q2' }, { id: 'q3' }] };
      assessmentState.initAssessment('module1', quizData);

      const newIndex = assessmentState.goToQuestion(2);
      expect(newIndex).toBe(2);
    });
  });

  describe('retryAssessment', () => {
    it('should clear current session for retry', () => {
      const quizData = { moduleName: 'Test', questions: [{ id: 'q1' }] };
      assessmentState.initAssessment('module1', quizData);

      const retryInfo = assessmentState.retryAssessment('module1');

      expect(retryInfo.moduleId).toBe('module1');
      expect(retryInfo.retry).toBe(true);
      expect(assessmentState.getCurrentAssessment()).toBeNull();
    });
  });

  describe('exportData and importData', () => {
    it('should export assessment data as JSON', () => {
      const quizData = { moduleName: 'Test', questions: [{ id: 'q1' }] };
      assessmentState.initAssessment('module1', quizData);
      assessmentState.completeAssessment({ score: 85, passed: true });

      const exported = assessmentState.exportData();
      const parsed = JSON.parse(exported);

      expect(parsed.module1).toBeDefined();
      expect(parsed.module1.length).toBe(1);
    });

    it('should import assessment data from JSON', () => {
      const importData = JSON.stringify({
        module1: [{ score: 90, passed: true, attemptNumber: 1 }]
      });

      const result = assessmentState.importData(importData);
      expect(result).toBe(true);

      const history = assessmentState.getModuleHistory('module1');
      expect(history.length).toBe(1);
      expect(history[0].score).toBe(90);
    });
  });
});
