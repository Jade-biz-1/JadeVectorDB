/**
 * Unit tests for quizScoring.js
 * Aligned with actual implementation
 */

import {
  gradeQuestion,
  calculateTotalScore,
  isPassing,
  analyzePerformance,
  getGradeLetter,
  getPerformanceLevel,
  calculateTimeMetrics,
  compareAttempts
} from '../lib/quizScoring';

describe('quizScoring', () => {
  describe('gradeQuestion', () => {
    it('should grade multiple choice question correctly', () => {
      const question = {
        id: 'q1',
        type: 'multiple-choice',
        points: 10,
        correctAnswer: 2
      };

      const correctResult = gradeQuestion(question, 2);
      expect(correctResult.isCorrect).toBe(true);
      expect(correctResult.earnedPoints).toBe(10);

      const incorrectResult = gradeQuestion(question, 0);
      expect(incorrectResult.isCorrect).toBe(false);
      expect(incorrectResult.earnedPoints).toBe(0);
    });

    it('should grade multiple answers question correctly', () => {
      const question = {
        id: 'q2',
        type: 'multiple-choice',
        multipleAnswers: true,
        points: 15,
        correctAnswer: [0, 2]  // Implementation uses correctAnswer (not correctAnswers)
      };

      // All correct (same answers, possibly different order)
      const perfectResult = gradeQuestion(question, [0, 2]);
      expect(perfectResult.isCorrect).toBe(true);
      expect(perfectResult.earnedPoints).toBe(15);

      // Wrong selection
      const wrongResult = gradeQuestion(question, [1, 3]);
      expect(wrongResult.isCorrect).toBe(false);
      expect(wrongResult.earnedPoints).toBe(0);
    });

    it('should grade true/false question correctly', () => {
      const question = {
        id: 'q3',
        type: 'true-false',
        points: 5,
        correctAnswer: true
      };

      const correctResult = gradeQuestion(question, true);
      expect(correctResult.isCorrect).toBe(true);
      expect(correctResult.earnedPoints).toBe(5);

      const incorrectResult = gradeQuestion(question, false);
      expect(incorrectResult.isCorrect).toBe(false);
      expect(incorrectResult.earnedPoints).toBe(0);
    });

    it('should grade fill-blank question correctly', () => {
      const question = {
        id: 'q4',
        type: 'fill-blank',
        points: 10,
        correctAnswer: 'vector'
      };

      // Exact match (case insensitive)
      const correctResult = gradeQuestion(question, 'Vector');
      expect(correctResult.isCorrect).toBe(true);
      expect(correctResult.earnedPoints).toBe(10);

      // Wrong answer
      const wrongResult = gradeQuestion(question, 'database');
      expect(wrongResult.isCorrect).toBe(false);
      expect(wrongResult.earnedPoints).toBe(0);
    });

    it('should handle null/undefined answers', () => {
      const question = {
        id: 'q5',
        type: 'multiple-choice',
        points: 10,
        correctAnswer: 1
      };

      const nullResult = gradeQuestion(question, null);
      expect(nullResult.isCorrect).toBe(false);
      expect(nullResult.earnedPoints).toBe(0);

      const undefinedResult = gradeQuestion(question, undefined);
      expect(undefinedResult.isCorrect).toBe(false);
      expect(undefinedResult.earnedPoints).toBe(0);
    });
  });

  describe('calculateTotalScore', () => {
    it('should calculate correct total score and percentage', () => {
      const gradedResults = [
        { points: 10, earnedPoints: 10, isCorrect: true },
        { points: 10, earnedPoints: 5, isCorrect: false },
        { points: 20, earnedPoints: 15, isCorrect: false }
      ];

      const result = calculateTotalScore(gradedResults);

      expect(result.earnedPoints).toBe(30);
      expect(result.totalPoints).toBe(40);
      expect(result.percentage).toBe(75);
      expect(result.correctCount).toBe(1);
      expect(result.totalQuestions).toBe(3);
    });

    it('should handle empty results', () => {
      const result = calculateTotalScore([]);

      expect(result.earnedPoints).toBe(0);
      expect(result.totalPoints).toBe(0);
      expect(result.percentage).toBe(0);
      expect(result.correctCount).toBe(0);
      expect(result.totalQuestions).toBe(0);
    });

    it('should calculate correct statistics', () => {
      const gradedResults = [
        { points: 10, earnedPoints: 10, isCorrect: true },
        { points: 10, earnedPoints: 0, isCorrect: false },
        { points: 20, earnedPoints: 20, isCorrect: true },
        { points: 5, earnedPoints: 0, isCorrect: false }
      ];

      const result = calculateTotalScore(gradedResults);

      expect(result.correctCount).toBe(2);
      expect(result.totalQuestions).toBe(4);
      expect(result.percentage).toBe(67); // 30/45 = 66.67%
    });
  });

  describe('isPassing', () => {
    it('should return true when score meets or exceeds minimum', () => {
      expect(isPassing(70, 70)).toBe(true);
      expect(isPassing(85, 70)).toBe(true);
      expect(isPassing(100, 70)).toBe(true);
    });

    it('should return false when score is below minimum', () => {
      expect(isPassing(69, 70)).toBe(false);
      expect(isPassing(50, 70)).toBe(false);
      expect(isPassing(0, 70)).toBe(false);
    });

    it('should use default minScore of 70', () => {
      expect(isPassing(70)).toBe(true);
      expect(isPassing(69)).toBe(false);
    });
  });

  describe('analyzePerformance', () => {
    it('should analyze performance by difficulty', () => {
      const gradedResults = [
        { difficulty: 'easy', type: 'multiple-choice', isCorrect: true },
        { difficulty: 'easy', type: 'multiple-choice', isCorrect: false },
        { difficulty: 'medium', type: 'multiple-choice', isCorrect: true },
        { difficulty: 'hard', type: 'multiple-choice', isCorrect: false }
      ];

      const analysis = analyzePerformance(gradedResults);

      expect(analysis.byDifficulty.easy.correct).toBe(1);
      expect(analysis.byDifficulty.easy.total).toBe(2);
      expect(analysis.byDifficulty.easy.percentage).toBe(50);
      expect(analysis.byDifficulty.medium.correct).toBe(1);
      expect(analysis.byDifficulty.medium.total).toBe(1);
      expect(analysis.byDifficulty.hard.correct).toBe(0);
      expect(analysis.byDifficulty.hard.total).toBe(1);
    });

    it('should analyze performance by type', () => {
      const gradedResults = [
        { type: 'multiple-choice', isCorrect: true },
        { type: 'multiple-choice', isCorrect: true },
        { type: 'true-false', isCorrect: false }
      ];

      const analysis = analyzePerformance(gradedResults);

      expect(analysis.byType['multiple-choice'].total).toBe(2);
      expect(analysis.byType['multiple-choice'].correct).toBe(2);
      expect(analysis.byType['true-false'].total).toBe(1);
      expect(analysis.byType['true-false'].correct).toBe(0);
    });

    it('should identify strengths with 80%+ performance', () => {
      const gradedResults = [
        { difficulty: 'easy', type: 'multiple-choice', isCorrect: true },
        { difficulty: 'easy', type: 'multiple-choice', isCorrect: true },
        { difficulty: 'easy', type: 'multiple-choice', isCorrect: true },
        { difficulty: 'easy', type: 'multiple-choice', isCorrect: true },
        { difficulty: 'easy', type: 'multiple-choice', isCorrect: true }
      ];

      const analysis = analyzePerformance(gradedResults);

      // Strengths contain messages about strong performance on easy questions
      expect(analysis.strengths.length).toBeGreaterThan(0);
      expect(analysis.strengths[0]).toContain('easy');
    });

    it('should identify weaknesses with <60% performance', () => {
      const gradedResults = [
        { difficulty: 'hard', type: 'code-challenge', isCorrect: false },
        { difficulty: 'hard', type: 'code-challenge', isCorrect: false },
        { difficulty: 'hard', type: 'code-challenge', isCorrect: false }
      ];

      const analysis = analyzePerformance(gradedResults);

      expect(analysis.weaknesses.length).toBeGreaterThan(0);
      expect(analysis.weaknesses[0]).toContain('hard');
    });

    it('should handle empty results', () => {
      const analysis = analyzePerformance([]);

      expect(analysis.byDifficulty).toBeDefined();
      expect(analysis.strengths).toEqual([]);
      expect(analysis.weaknesses).toEqual([]);
      // Note: recommendations may still be generated based on threshold logic
      expect(analysis.recommendations).toBeDefined();
    });
  });

  describe('getGradeLetter', () => {
    it('should return correct letter grades', () => {
      // Implementation uses simple grades: A, B, C, D, F
      expect(getGradeLetter(95)).toBe('A');
      expect(getGradeLetter(90)).toBe('A');
      expect(getGradeLetter(85)).toBe('B');
      expect(getGradeLetter(80)).toBe('B');
      expect(getGradeLetter(75)).toBe('C');
      expect(getGradeLetter(70)).toBe('C');
      expect(getGradeLetter(65)).toBe('D');
      expect(getGradeLetter(60)).toBe('D');
      expect(getGradeLetter(55)).toBe('F');
      expect(getGradeLetter(50)).toBe('F');
    });

    it('should handle edge cases', () => {
      expect(getGradeLetter(100)).toBe('A');
      expect(getGradeLetter(0)).toBe('F');
    });

    it('should handle boundary values', () => {
      expect(getGradeLetter(89)).toBe('B');  // Just below A threshold
      expect(getGradeLetter(79)).toBe('C');  // Just below B threshold
      expect(getGradeLetter(69)).toBe('D');  // Just below C threshold
      expect(getGradeLetter(59)).toBe('F');  // Just below D threshold
    });
  });

  describe('getPerformanceLevel', () => {
    it('should return correct performance level objects', () => {
      // Implementation returns objects with level, description, and color
      const excellent = getPerformanceLevel(95);
      expect(excellent.level).toBe('Excellent');
      expect(excellent.description).toBeDefined();
      expect(excellent.color).toBe('#22c55e');

      const veryGood = getPerformanceLevel(85);
      expect(veryGood.level).toBe('Very Good');

      const good = getPerformanceLevel(75);
      expect(good.level).toBe('Good');

      const fair = getPerformanceLevel(65);
      expect(fair.level).toBe('Fair');

      const needsImprovement = getPerformanceLevel(50);
      expect(needsImprovement.level).toBe('Needs Improvement');
    });

    it('should handle edge cases', () => {
      const perfect = getPerformanceLevel(100);
      expect(perfect.level).toBe('Excellent');

      const zero = getPerformanceLevel(0);
      expect(zero.level).toBe('Needs Improvement');
    });

    it('should include appropriate colors', () => {
      expect(getPerformanceLevel(95).color).toBe('#22c55e'); // green
      expect(getPerformanceLevel(85).color).toBe('#3b82f6'); // blue
      expect(getPerformanceLevel(75).color).toBe('#eab308'); // yellow
      expect(getPerformanceLevel(65).color).toBe('#f97316'); // orange
      expect(getPerformanceLevel(50).color).toBe('#ef4444'); // red
    });
  });

  describe('calculateTimeMetrics', () => {
    it('should calculate time metrics correctly', () => {
      const startTime = 1000000;
      const endTime = 1000000 + (5 * 60 * 1000) + (30 * 1000); // 5:30

      const metrics = calculateTimeMetrics(startTime, endTime);

      expect(metrics.minutes).toBe(5);
      expect(metrics.seconds).toBe(30);
      expect(metrics.formatted).toBe('5:30');
      expect(metrics.totalSeconds).toBe(330);
    });

    it('should handle zero time', () => {
      const metrics = calculateTimeMetrics(1000, 1000);

      expect(metrics.minutes).toBe(0);
      expect(metrics.seconds).toBe(0);
      expect(metrics.formatted).toBe('0:00');
    });
  });

  describe('compareAttempts', () => {
    it('should compare two attempts correctly', () => {
      const attempt1 = { score: 70, timeSpent: 300000 };
      const attempt2 = { score: 85, timeSpent: 250000 };

      const comparison = compareAttempts(attempt1, attempt2);

      expect(comparison.scoreDifference).toBe(15);
      expect(comparison.scoreImprovement).toBe(true);
      expect(comparison.fasterCompletion).toBe(true);
    });

    it('should handle declining performance', () => {
      const attempt1 = { score: 90, timeSpent: 200000 };
      const attempt2 = { score: 75, timeSpent: 300000 };

      const comparison = compareAttempts(attempt1, attempt2);

      expect(comparison.scoreDifference).toBe(-15);
      expect(comparison.scoreImprovement).toBe(false);
      expect(comparison.fasterCompletion).toBe(false);
    });
  });
});
