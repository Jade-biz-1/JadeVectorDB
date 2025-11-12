import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { AssessmentEngine } from '../lib/assessmentEngine';

describe('AssessmentEngine', () => {
  let engine;

  beforeEach(() => {
    engine = new AssessmentEngine();
    // Clear localStorage before each test
    localStorage.clear();
  });

  afterEach(() => {
    localStorage.clear();
  });

  describe('validateAnswer', () => {
    it('should validate multiple-choice answers correctly', () => {
      const question = {
        type: 'multiple-choice',
        correctAnswer: 2
      };

      expect(engine.validateAnswer(question, 2)).toBe(true);
      expect(engine.validateAnswer(question, 0)).toBe(false);
      expect(engine.validateAnswer(question, 1)).toBe(false);
    });

    it('should validate code-completion answers with flexible matching', () => {
      const question = {
        type: 'code-completion',
        correctAnswer: 'client.create_collection("my_collection")'
      };

      // Exact match
      expect(engine.validateAnswer(question, 'client.create_collection("my_collection")')).toBe(true);

      // Different whitespace
      expect(engine.validateAnswer(question, 'client.create_collection( "my_collection" )')).toBe(true);

      // Wrong answer
      expect(engine.validateAnswer(question, 'client.delete_collection("my_collection")')).toBe(false);
    });

    it('should handle debugging questions', () => {
      const question = {
        type: 'debugging',
        correctAnswer: 'await'
      };

      expect(engine.validateAnswer(question, 'response = await client.search(...)')).toBe(true);
      expect(engine.validateAnswer(question, 'response = client.search(...)')).toBe(false);
    });

    it('should validate scenario-based questions', () => {
      const question = {
        type: 'scenario-based',
        correctAnswer: 1
      };

      expect(engine.validateAnswer(question, 1)).toBe(true);
      expect(engine.validateAnswer(question, 0)).toBe(false);
    });
  });

  describe('calculateScore', () => {
    const questions = [
      { id: 'q1', type: 'multiple-choice', correctAnswer: 1, points: 10, explanation: 'Test' },
      { id: 'q2', type: 'multiple-choice', correctAnswer: 2, points: 15, explanation: 'Test' },
      { id: 'q3', type: 'multiple-choice', correctAnswer: 0, points: 20, explanation: 'Test' }
    ];

    it('should calculate perfect score correctly', () => {
      const userAnswers = [1, 2, 0];
      const result = engine.calculateScore(questions, userAnswers);

      expect(result.earnedPoints).toBe(45);
      expect(result.totalPoints).toBe(45);
      expect(result.percentage).toBe(100);
      expect(result.passed).toBe(true);
    });

    it('should calculate partial score correctly', () => {
      const userAnswers = [1, 0, 0]; // First and third correct
      const result = engine.calculateScore(questions, userAnswers);

      expect(result.earnedPoints).toBe(30); // 10 + 20
      expect(result.totalPoints).toBe(45);
      expect(result.percentage).toBeCloseTo(66.7, 1);
      expect(result.passed).toBe(false);
    });

    it('should calculate zero score correctly', () => {
      const userAnswers = [0, 0, 1]; // All wrong
      const result = engine.calculateScore(questions, userAnswers);

      expect(result.earnedPoints).toBe(0);
      expect(result.totalPoints).toBe(45);
      expect(result.percentage).toBe(0);
      expect(result.passed).toBe(false);
    });

    it('should include detailed results for each question', () => {
      const userAnswers = [1, 0, 0];
      const result = engine.calculateScore(questions, userAnswers);

      expect(result.results).toHaveLength(3);
      expect(result.results[0].isCorrect).toBe(true);
      expect(result.results[1].isCorrect).toBe(false);
      expect(result.results[2].isCorrect).toBe(true);
    });
  });

  describe('Progress Management', () => {
    it('should save progress to localStorage', () => {
      const progress = engine.saveProgress('module-1', 2, [1, 2, null], Date.now());
      expect(progress).toBe(true);

      const saved = localStorage.getItem(engine.STORAGE_KEY);
      expect(saved).toBeTruthy();

      const parsed = JSON.parse(saved);
      expect(parsed.moduleId).toBe('module-1');
      expect(parsed.currentQuestionIndex).toBe(2);
    });

    it('should load saved progress from localStorage', () => {
      const startTime = Date.now();
      engine.saveProgress('module-1', 3, [1, 2, 0], startTime);

      const loaded = engine.loadProgress();
      expect(loaded).toBeTruthy();
      expect(loaded.moduleId).toBe('module-1');
      expect(loaded.currentQuestionIndex).toBe(3);
      expect(loaded.userAnswers).toEqual([1, 2, 0]);
    });

    it('should clear progress', () => {
      engine.saveProgress('module-1', 2, [1, 2], Date.now());
      expect(localStorage.getItem(engine.STORAGE_KEY)).toBeTruthy();

      engine.clearProgress();
      expect(localStorage.getItem(engine.STORAGE_KEY)).toBeNull();
    });

    it('should return null when no progress exists', () => {
      const loaded = engine.loadProgress();
      expect(loaded).toBeNull();
    });
  });

  describe('Results Management', () => {
    const mockScoreData = {
      earnedPoints: 80,
      totalPoints: 100,
      percentage: 80,
      passed: true,
      results: []
    };

    it('should save quiz results', () => {
      const saved = engine.saveResults('module-1', mockScoreData, 300);
      expect(saved).toBe(true);

      const results = engine.loadModuleResults('module-1');
      expect(results).toHaveLength(1);
      expect(results[0].percentage).toBe(80);
      expect(results[0].timeSpent).toBe(300);
    });

    it('should load results for specific module', () => {
      engine.saveResults('module-1', mockScoreData, 300);
      engine.saveResults('module-2', { ...mockScoreData, percentage: 90 }, 250);

      const module1Results = engine.loadModuleResults('module-1');
      const module2Results = engine.loadModuleResults('module-2');

      expect(module1Results).toHaveLength(1);
      expect(module2Results).toHaveLength(1);
      expect(module1Results[0].percentage).toBe(80);
      expect(module2Results[0].percentage).toBe(90);
    });

    it('should track multiple attempts for same module', () => {
      engine.saveResults('module-1', { ...mockScoreData, percentage: 70 }, 300);
      engine.saveResults('module-1', { ...mockScoreData, percentage: 85 }, 280);
      engine.saveResults('module-1', { ...mockScoreData, percentage: 95 }, 250);

      const results = engine.loadModuleResults('module-1');
      expect(results).toHaveLength(3);
    });

    it('should get best score for module', () => {
      engine.saveResults('module-1', { ...mockScoreData, percentage: 70 }, 300);
      engine.saveResults('module-1', { ...mockScoreData, percentage: 85 }, 280);
      engine.saveResults('module-1', { ...mockScoreData, percentage: 95 }, 250);

      const bestScore = engine.getBestScore('module-1');
      expect(bestScore.percentage).toBe(95);
    });

    it('should return null for best score when no results exist', () => {
      const bestScore = engine.getBestScore('nonexistent');
      expect(bestScore).toBeNull();
    });
  });

  describe('Statistics', () => {
    beforeEach(() => {
      // Add sample results
      engine.saveResults('module-1', {
        earnedPoints: 70,
        totalPoints: 100,
        percentage: 70,
        passed: true,
        results: []
      }, 300);

      engine.saveResults('module-1', {
        earnedPoints: 85,
        totalPoints: 100,
        percentage: 85,
        passed: true,
        results: []
      }, 250);

      engine.saveResults('module-2', {
        earnedPoints: 60,
        totalPoints: 100,
        percentage: 60,
        passed: false,
        results: []
      }, 400);
    });

    it('should calculate module statistics correctly', () => {
      const stats = engine.getStatistics('module-1');

      expect(stats.attempts).toBe(2);
      expect(stats.bestScore).toBe(85);
      expect(stats.averageScore).toBe(77.5);
      expect(stats.passed).toBe(true);
      expect(stats.totalTimeSpent).toBe(550);
    });

    it('should calculate overall statistics', () => {
      const stats = engine.getStatistics();

      expect(stats.totalQuizzes).toBe(3);
      expect(stats.passedQuizzes).toBe(2);
      expect(stats.totalTimeSpent).toBe(950);
      expect(stats.moduleStats['module-1']).toBeDefined();
      expect(stats.moduleStats['module-2']).toBeDefined();
    });

    it('should handle empty statistics', () => {
      localStorage.clear();
      const stats = engine.getStatistics('module-1');

      expect(stats.attempts).toBe(0);
      expect(stats.bestScore).toBe(0);
      expect(stats.passed).toBe(false);
    });
  });

  describe('Time Management', () => {
    it('should format time correctly', () => {
      expect(engine.formatTime(45)).toBe('45s');
      expect(engine.formatTime(60)).toBe('1m 0s');
      expect(engine.formatTime(125)).toBe('2m 5s');
      expect(engine.formatTime(600)).toBe('10m 0s');
    });

    it('should check if time expired', () => {
      const startTime = Date.now() - 610000; // 10 minutes 10 seconds ago
      const timeLimit = 600; // 10 minutes

      expect(engine.isTimeExpired(startTime, timeLimit)).toBe(true);
    });

    it('should check if time not expired', () => {
      const startTime = Date.now() - 300000; // 5 minutes ago
      const timeLimit = 600; // 10 minutes

      expect(engine.isTimeExpired(startTime, timeLimit)).toBe(false);
    });

    it('should calculate remaining time', () => {
      const startTime = Date.now() - 300000; // 5 minutes ago
      const timeLimit = 600; // 10 minutes

      const remaining = engine.getRemainingTime(startTime, timeLimit);
      expect(remaining).toBeGreaterThan(290);
      expect(remaining).toBeLessThanOrEqual(300);
    });
  });

  describe('Feedback Generation', () => {
    it('should generate excellent feedback for 90%+', () => {
      const feedback = engine.generateFeedback(95, 'module-1');

      expect(feedback.level).toBe('excellent');
      expect(feedback.message).toContain('Outstanding');
      expect(feedback.suggestions).toBeInstanceOf(Array);
      expect(feedback.suggestions.length).toBeGreaterThan(0);
    });

    it('should generate good feedback for 70-89%', () => {
      const feedback = engine.generateFeedback(75, 'module-1');

      expect(feedback.level).toBe('good');
      expect(feedback.message).toContain('Great job');
      expect(feedback.suggestions).toBeInstanceOf(Array);
    });

    it('should generate needs-improvement feedback for 50-69%', () => {
      const feedback = engine.generateFeedback(60, 'module-1');

      expect(feedback.level).toBe('needs-improvement');
      expect(feedback.message).toContain('getting there');
      expect(feedback.suggestions).toBeInstanceOf(Array);
    });

    it('should generate needs-review feedback for below 50%', () => {
      const feedback = engine.generateFeedback(35, 'module-1');

      expect(feedback.level).toBe('needs-review');
      expect(feedback.message).toContain('review');
      expect(feedback.suggestions).toBeInstanceOf(Array);
    });
  });

  describe('Code Validation', () => {
    it('should extract key elements from code', () => {
      const code = 'client.create_collection("test").add_vectors(vectors)';
      const elements = engine.extractKeyElements(code);

      expect(elements).toContain('create_collection');
      expect(elements).toContain('add_vectors');
    });

    it('should validate code completion with key elements', () => {
      const question = {
        type: 'code-completion',
        correctAnswer: 'response = await client.search(query_vector, top_k=5)'
      };

      // Contains all key elements
      const userAnswer = 'response = await client.search(query_vector, top_k=5)';
      expect(engine.validateAnswer(question, userAnswer)).toBe(true);

      // Missing key element
      const wrongAnswer = 'response = client.get(query_vector)';
      expect(engine.validateAnswer(question, wrongAnswer)).toBe(false);
    });
  });

  describe('Export and Clear', () => {
    beforeEach(() => {
      engine.saveResults('module-1', {
        earnedPoints: 80,
        totalPoints: 100,
        percentage: 80,
        passed: true,
        results: []
      }, 300);
    });

    it('should export results for specific module', () => {
      const exported = engine.exportResults('module-1');

      expect(exported.exportDate).toBeDefined();
      expect(exported.version).toBe('1.0');
      expect(exported.results['module-1']).toBeDefined();
      expect(exported.statistics).toBeDefined();
    });

    it('should export all results', () => {
      engine.saveResults('module-2', {
        earnedPoints: 90,
        totalPoints: 100,
        percentage: 90,
        passed: true,
        results: []
      }, 250);

      const exported = engine.exportResults();

      expect(exported.results['module-1']).toBeDefined();
      expect(exported.results['module-2']).toBeDefined();
    });

    it('should clear all results', () => {
      engine.clearAllResults();

      const results = engine.loadAllResults();
      expect(Object.keys(results)).toHaveLength(0);
    });
  });
});
