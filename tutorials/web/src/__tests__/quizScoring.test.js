/**
 * Unit tests for quizScoring.js
 */

import {
  gradeQuestion,
  calculateTotalScore,
  isPassing,
  analyzePerformance,
  getGradeLetter,
  getPerformanceLevel
} from '../lib/quizScoring';

describe('quizScoring', () => {
  describe('gradeQuestion', () => {
    it('should grade multiple choice question correctly', () => {
      const question = {
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
        type: 'multiple-choice',
        multipleAnswers: true,
        points: 15,
        correctAnswers: [0, 2]
      };

      // All correct
      const perfectResult = gradeQuestion(question, [0, 2]);
      expect(perfectResult.isCorrect).toBe(true);
      expect(perfectResult.earnedPoints).toBe(15);

      // Partial correct
      const partialResult = gradeQuestion(question, [0]);
      expect(partialResult.isCorrect).toBe(false);
      expect(partialResult.earnedPoints).toBe(7.5); // 50% partial credit

      // All wrong
      const wrongResult = gradeQuestion(question, [1, 3]);
      expect(wrongResult.isCorrect).toBe(false);
      expect(wrongResult.earnedPoints).toBe(0);
    });

    it('should grade true/false question correctly', () => {
      const question = {
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

    it('should grade code challenge with test cases', () => {
      const question = {
        type: 'code-challenge',
        points: 20,
        testCases: [
          { input: '1', expected: '2' },
          { input: '2', expected: '4' },
          { input: '3', expected: '6' }
        ]
      };

      // All test cases pass
      const perfectResult = gradeQuestion(question, {
        passedTests: 3,
        totalTests: 3
      });
      expect(perfectResult.isCorrect).toBe(true);
      expect(perfectResult.earnedPoints).toBe(20);

      // Partial test cases pass
      const partialResult = gradeQuestion(question, {
        passedTests: 2,
        totalTests: 3
      });
      expect(partialResult.isCorrect).toBe(false);
      expect(partialResult.earnedPoints).toBeCloseTo(13.33, 1);

      // No test cases pass
      const failResult = gradeQuestion(question, {
        passedTests: 0,
        totalTests: 3
      });
      expect(failResult.isCorrect).toBe(false);
      expect(failResult.earnedPoints).toBe(0);
    });
  });

  describe('calculateTotalScore', () => {
    it('should calculate correct total score and percentage', () => {
      const gradedResults = [
        { earnedPoints: 10, totalPoints: 10, isCorrect: true },
        { earnedPoints: 5, totalPoints: 10, isCorrect: false },
        { earnedPoints: 15, totalPoints: 20, isCorrect: false }
      ];

      const result = calculateTotalScore(gradedResults);

      expect(result.earnedPoints).toBe(30);
      expect(result.totalPoints).toBe(40);
      expect(result.percentage).toBe(75);
    });

    it('should handle empty results', () => {
      const result = calculateTotalScore([]);

      expect(result.earnedPoints).toBe(0);
      expect(result.totalPoints).toBe(0);
      expect(result.percentage).toBe(0);
    });

    it('should calculate correct statistics', () => {
      const gradedResults = [
        { earnedPoints: 10, totalPoints: 10, isCorrect: true },
        { earnedPoints: 0, totalPoints: 10, isCorrect: false },
        { earnedPoints: 20, totalPoints: 20, isCorrect: true },
        { earnedPoints: 0, totalPoints: 5, isCorrect: false }
      ];

      const result = calculateTotalScore(gradedResults);

      expect(result.correctCount).toBe(2);
      expect(result.incorrectCount).toBe(2);
      expect(result.totalQuestions).toBe(4);
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
  });

  describe('analyzePerformance', () => {
    it('should analyze performance by difficulty', () => {
      const gradedResults = [
        {
          question: { difficulty: 'easy' },
          isCorrect: true,
          earnedPoints: 10,
          totalPoints: 10
        },
        {
          question: { difficulty: 'easy' },
          isCorrect: false,
          earnedPoints: 0,
          totalPoints: 10
        },
        {
          question: { difficulty: 'medium' },
          isCorrect: true,
          earnedPoints: 15,
          totalPoints: 15
        },
        {
          question: { difficulty: 'hard' },
          isCorrect: false,
          earnedPoints: 0,
          totalPoints: 20
        }
      ];

      const analysis = analyzePerformance(gradedResults);

      expect(analysis.byDifficulty.easy.correct).toBe(1);
      expect(analysis.byDifficulty.easy.total).toBe(2);
      expect(analysis.byDifficulty.medium.correct).toBe(1);
      expect(analysis.byDifficulty.medium.total).toBe(1);
      expect(analysis.byDifficulty.hard.correct).toBe(0);
      expect(analysis.byDifficulty.hard.total).toBe(1);
    });

    it('should identify strengths and weaknesses', () => {
      const gradedResults = [
        {
          question: { difficulty: 'easy', type: 'multiple-choice' },
          isCorrect: true,
          earnedPoints: 10,
          totalPoints: 10
        },
        {
          question: { difficulty: 'easy', type: 'multiple-choice' },
          isCorrect: true,
          earnedPoints: 10,
          totalPoints: 10
        },
        {
          question: { difficulty: 'hard', type: 'code-challenge' },
          isCorrect: false,
          earnedPoints: 0,
          totalPoints: 20
        },
        {
          question: { difficulty: 'hard', type: 'code-challenge' },
          isCorrect: false,
          earnedPoints: 0,
          totalPoints: 20
        }
      ];

      const analysis = analyzePerformance(gradedResults);

      expect(analysis.strengths).toContain('Easy questions');
      expect(analysis.weaknesses).toContain('Hard questions');
    });

    it('should handle empty results', () => {
      const analysis = analyzePerformance([]);

      expect(analysis.byDifficulty).toBeDefined();
      expect(analysis.strengths).toEqual([]);
      expect(analysis.weaknesses).toEqual([]);
    });
  });

  describe('getGradeLetter', () => {
    it('should return correct letter grades', () => {
      expect(getGradeLetter(95)).toBe('A+');
      expect(getGradeLetter(92)).toBe('A');
      expect(getGradeLetter(88)).toBe('A-');
      expect(getGradeLetter(85)).toBe('B+');
      expect(getGradeLetter(82)).toBe('B');
      expect(getGradeLetter(78)).toBe('B-');
      expect(getGradeLetter(75)).toBe('C+');
      expect(getGradeLetter(72)).toBe('C');
      expect(getGradeLetter(68)).toBe('C-');
      expect(getGradeLetter(65)).toBe('D+');
      expect(getGradeLetter(62)).toBe('D');
      expect(getGradeLetter(58)).toBe('D-');
      expect(getGradeLetter(50)).toBe('F');
    });

    it('should handle edge cases', () => {
      expect(getGradeLetter(100)).toBe('A+');
      expect(getGradeLetter(0)).toBe('F');
    });
  });

  describe('getPerformanceLevel', () => {
    it('should return correct performance levels', () => {
      expect(getPerformanceLevel(95)).toBe('Excellent');
      expect(getPerformanceLevel(85)).toBe('Very Good');
      expect(getPerformanceLevel(75)).toBe('Good');
      expect(getPerformanceLevel(65)).toBe('Fair');
      expect(getPerformanceLevel(50)).toBe('Needs Improvement');
    });

    it('should handle edge cases', () => {
      expect(getPerformanceLevel(100)).toBe('Excellent');
      expect(getPerformanceLevel(0)).toBe('Needs Improvement');
    });
  });
});
