/**
 * Unit tests for assessmentState.js
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
      expect(session.quizData).toEqual(quizData);
      expect(session.answers).toEqual({});
      expect(session.startTime).toBeDefined();
      expect(session.completed).toBe(false);
    });

    it('should generate unique session ID', () => {
      const quizData = { moduleId: 'module1', questions: [] };
      const session1 = assessmentState.initAssessment('module1', quizData);
      const session2 = assessmentState.initAssessment('module1', quizData);

      expect(session1.sessionId).not.toBe(session2.sessionId);
    });
  });

  describe('saveAnswer', () => {
    it('should save an answer to current session', () => {
      const quizData = { moduleId: 'module1', questions: [] };
      assessmentState.initAssessment('module1', quizData);

      assessmentState.saveAnswer('q1', 0);
      const session = assessmentState.getCurrentSession();

      expect(session.answers.q1).toBe(0);
    });

    it('should update existing answer', () => {
      const quizData = { moduleId: 'module1', questions: [] };
      assessmentState.initAssessment('module1', quizData);

      assessmentState.saveAnswer('q1', 0);
      assessmentState.saveAnswer('q1', 2);
      const session = assessmentState.getCurrentSession();

      expect(session.answers.q1).toBe(2);
    });
  });

  describe('completeAssessment', () => {
    it('should mark assessment as complete', () => {
      const quizData = { moduleId: 'module1', questions: [] };
      assessmentState.initAssessment('module1', quizData);

      const result = {
        score: 85,
        passed: true,
        totalPoints: 100,
        earnedPoints: 85
      };

      assessmentState.completeAssessment(result);
      const session = assessmentState.getCurrentSession();

      expect(session.completed).toBe(true);
      expect(session.result).toEqual(result);
      expect(session.endTime).toBeDefined();
    });

    it('should add to assessment history', () => {
      const quizData = { moduleId: 'module1', questions: [] };
      assessmentState.initAssessment('module1', quizData);

      const result = { score: 85, passed: true };
      assessmentState.completeAssessment(result);

      const history = assessmentState.getModuleHistory('module1');
      expect(history.length).toBe(1);
      expect(history[0].result.score).toBe(85);
    });
  });

  describe('getModuleHistory', () => {
    it('should return empty array for module with no history', () => {
      const history = assessmentState.getModuleHistory('module99');
      expect(history).toEqual([]);
    });

    it('should return all attempts for a module', () => {
      const quizData = { moduleId: 'module1', questions: [] };

      // First attempt
      assessmentState.initAssessment('module1', quizData);
      assessmentState.completeAssessment({ score: 60, passed: false });

      // Second attempt
      assessmentState.initAssessment('module1', quizData);
      assessmentState.completeAssessment({ score: 80, passed: true });

      const history = assessmentState.getModuleHistory('module1');
      expect(history.length).toBe(2);
      expect(history[0].result.score).toBe(60);
      expect(history[1].result.score).toBe(80);
    });
  });

  describe('getBestScore', () => {
    it('should return 0 for module with no attempts', () => {
      const bestScore = assessmentState.getBestScore('module99');
      expect(bestScore).toBe(0);
    });

    it('should return highest score from multiple attempts', () => {
      const quizData = { moduleId: 'module1', questions: [] };

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
      const quizData = { moduleId: 'module1', questions: [] };

      assessmentState.initAssessment('module1', quizData);
      assessmentState.completeAssessment({ score: 60, passed: false });

      expect(assessmentState.hasPassedModule('module1')).toBe(false);
    });

    it('should return true if at least one attempt passed', () => {
      const quizData = { moduleId: 'module1', questions: [] };

      assessmentState.initAssessment('module1', quizData);
      assessmentState.completeAssessment({ score: 60, passed: false });

      assessmentState.initAssessment('module1', quizData);
      assessmentState.completeAssessment({ score: 80, passed: true });

      expect(assessmentState.hasPassedModule('module1')).toBe(true);
    });
  });

  describe('getOverallProgress', () => {
    it('should calculate correct progress stats', () => {
      const quizData1 = { moduleId: 'module1', questions: [] };
      const quizData2 = { moduleId: 'module2', questions: [] };

      // Complete module1
      assessmentState.initAssessment('module1', quizData1);
      assessmentState.completeAssessment({ score: 80, passed: true });

      // Complete module2
      assessmentState.initAssessment('module2', quizData2);
      assessmentState.completeAssessment({ score: 90, passed: true });

      const progress = assessmentState.getOverallProgress();

      expect(progress.totalModules).toBeGreaterThan(0);
      expect(progress.completedModules).toBe(2);
      expect(progress.moduleScores.module1).toBe(80);
      expect(progress.moduleScores.module2).toBe(90);
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
      const quizData = { moduleId: 'module1', questions: [] };
      assessmentState.initAssessment('module1', quizData);
      assessmentState.completeAssessment({ score: 80, passed: true });

      assessmentState.clearHistory();

      const history = assessmentState.getModuleHistory('module1');
      expect(history).toEqual([]);
    });
  });

  describe('clearModuleHistory', () => {
    it('should clear history for specific module only', () => {
      const quizData1 = { moduleId: 'module1', questions: [] };
      const quizData2 = { moduleId: 'module2', questions: [] };

      assessmentState.initAssessment('module1', quizData1);
      assessmentState.completeAssessment({ score: 80, passed: true });

      assessmentState.initAssessment('module2', quizData2);
      assessmentState.completeAssessment({ score: 90, passed: true });

      assessmentState.clearModuleHistory('module1');

      expect(assessmentState.getModuleHistory('module1')).toEqual([]);
      expect(assessmentState.getModuleHistory('module2').length).toBe(1);
    });
  });
});
