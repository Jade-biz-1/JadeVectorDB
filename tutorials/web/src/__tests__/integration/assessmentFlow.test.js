/**
 * Integration tests for complete assessment flow
 */

import assessmentState from '../../lib/assessmentState';
import { gradeQuestion, calculateTotalScore } from '../../lib/quizScoring';
import { checkAchievements, clearAchievements } from '../../lib/achievementLogic';
import { evaluateReadiness } from '../../lib/readinessEvaluation';

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

describe('Assessment Flow Integration', () => {
  beforeEach(() => {
    localStorage.clear();
    clearAchievements();
  });

  describe('Complete Module Assessment Flow', () => {
    it('should complete full assessment workflow', () => {
      // 1. Initialize assessment
      const quizData = {
        moduleId: 'module1',
        moduleName: 'Getting Started',
        passingScore: 70,
        questions: [
          {
            id: 'q1',
            type: 'multiple-choice',
            question: 'What is a vector database?',
            points: 10,
            difficulty: 'easy',
            correctAnswer: 0,
            options: ['Database for vectors', 'SQL database', 'NoSQL database', 'Graph database']
          },
          {
            id: 'q2',
            type: 'true-false',
            question: 'Vector databases use similarity search',
            points: 10,
            difficulty: 'easy',
            correctAnswer: true
          },
          {
            id: 'q3',
            type: 'multiple-choice',
            question: 'Which algorithm is used for similarity search?',
            points: 15,
            difficulty: 'medium',
            correctAnswer: 1,
            options: ['Binary search', 'HNSW', 'Quick sort', 'Hash table']
          }
        ]
      };

      const session = assessmentState.initAssessment('module1', quizData);

      expect(session.moduleId).toBe('module1');
      expect(session.answers).toEqual({});

      // 2. Answer questions
      assessmentState.saveAnswer('q1', 0); // Correct
      assessmentState.saveAnswer('q2', true); // Correct
      assessmentState.saveAnswer('q3', 1); // Correct

      // 3. Grade answers
      const gradedResults = quizData.questions.map(question => {
        const currentSession = assessmentState.getCurrentSession();
        const userAnswer = currentSession.answers[question.id];
        return gradeQuestion(question, userAnswer);
      });

      expect(gradedResults.length).toBe(3);
      expect(gradedResults.every(r => r.isCorrect)).toBe(true);

      // 4. Calculate total score
      const totalScore = calculateTotalScore(gradedResults);

      expect(totalScore.percentage).toBe(100);
      expect(totalScore.earnedPoints).toBe(35);
      expect(totalScore.totalPoints).toBe(35);

      // 5. Determine pass/fail
      const passed = totalScore.percentage >= quizData.passingScore;
      expect(passed).toBe(true);

      // 6. Complete assessment
      const result = {
        score: totalScore.percentage,
        passed,
        totalPoints: totalScore.totalPoints,
        earnedPoints: totalScore.earnedPoints,
        gradedResults
      };

      assessmentState.completeAssessment(result);

      // 7. Verify assessment is saved
      const history = assessmentState.getModuleHistory('module1');
      expect(history.length).toBe(1);
      expect(history[0].result.score).toBe(100);

      // 8. Check for achievements
      const timeSpent = Date.now() - session.startTime;
      const achievementContext = {
        moduleId: 'module1',
        score: 100,
        passed: true,
        timeSpent
      };

      const unlockedAchievements = checkAchievements(achievementContext);

      // Should unlock at least module completion achievement
      expect(unlockedAchievements.length).toBeGreaterThan(0);

      // Should include perfect score achievement
      const perfectScoreAchievement = unlockedAchievements.find(a =>
        a.condition?.type === 'perfect_score'
      );
      expect(perfectScoreAchievement).toBeDefined();
    });

    it('should handle failed assessment with retry', () => {
      const quizData = {
        moduleId: 'module2',
        moduleName: 'Vector Search',
        passingScore: 70,
        questions: [
          {
            id: 'q1',
            type: 'multiple-choice',
            question: 'Question 1',
            points: 100,
            difficulty: 'hard',
            correctAnswer: 0,
            options: ['A', 'B', 'C', 'D']
          }
        ]
      };

      // First attempt - fail
      assessmentState.initAssessment('module2', quizData);
      assessmentState.saveAnswer('q1', 2); // Wrong answer

      const gradedResults1 = quizData.questions.map(question => {
        const currentSession = assessmentState.getCurrentSession();
        const userAnswer = currentSession.answers[question.id];
        return gradeQuestion(question, userAnswer);
      });

      const totalScore1 = calculateTotalScore(gradedResults1);
      const passed1 = totalScore1.percentage >= quizData.passingScore;

      expect(passed1).toBe(false);

      assessmentState.completeAssessment({
        score: totalScore1.percentage,
        passed: passed1,
        gradedResults: gradedResults1
      });

      expect(assessmentState.hasPassedModule('module2')).toBe(false);

      // Second attempt - pass
      assessmentState.initAssessment('module2', quizData);
      assessmentState.saveAnswer('q1', 0); // Correct answer

      const gradedResults2 = quizData.questions.map(question => {
        const currentSession = assessmentState.getCurrentSession();
        const userAnswer = currentSession.answers[question.id];
        return gradeQuestion(question, userAnswer);
      });

      const totalScore2 = calculateTotalScore(gradedResults2);
      const passed2 = totalScore2.percentage >= quizData.passingScore;

      expect(passed2).toBe(true);

      assessmentState.completeAssessment({
        score: totalScore2.percentage,
        passed: passed2,
        gradedResults: gradedResults2
      });

      // Verify module is now passed
      expect(assessmentState.hasPassedModule('module2')).toBe(true);

      // Verify history shows both attempts
      const history = assessmentState.getModuleHistory('module2');
      expect(history.length).toBe(2);
      expect(history[0].result.passed).toBe(false);
      expect(history[1].result.passed).toBe(true);

      // Best score should be the passing one
      const bestScore = assessmentState.getBestScore('module2');
      expect(bestScore).toBe(100);
    });
  });

  describe('Multi-Module Progress', () => {
    it('should track progress across multiple modules', () => {
      const modules = ['module1', 'module2', 'module3'];

      modules.forEach((moduleId, index) => {
        const quizData = {
          moduleId,
          moduleName: `Module ${index + 1}`,
          passingScore: 70,
          questions: [
            {
              id: 'q1',
              type: 'multiple-choice',
              question: 'Test question',
              points: 100,
              difficulty: 'easy',
              correctAnswer: 0,
              options: ['A', 'B', 'C', 'D']
            }
          ]
        };

        assessmentState.initAssessment(moduleId, quizData);
        assessmentState.saveAnswer('q1', 0);

        const gradedResults = quizData.questions.map(question => {
          const currentSession = assessmentState.getCurrentSession();
          const userAnswer = currentSession.answers[question.id];
          return gradeQuestion(question, userAnswer);
        });

        const totalScore = calculateTotalScore(gradedResults);

        assessmentState.completeAssessment({
          score: totalScore.percentage,
          passed: true,
          gradedResults
        });
      });

      // Check overall progress
      const progress = assessmentState.getOverallProgress();

      expect(progress.completedModules).toBe(3);
      expect(progress.moduleScores.module1).toBe(100);
      expect(progress.moduleScores.module2).toBe(100);
      expect(progress.moduleScores.module3).toBe(100);
      expect(progress.averageScore).toBe(100);
    });
  });

  describe('Readiness Assessment Integration', () => {
    it('should evaluate readiness after completing all modules', () => {
      // Complete all 6 modules with varying scores
      const moduleScores = [85, 90, 75, 80, 95, 88];

      moduleScores.forEach((score, index) => {
        const moduleId = `module${index + 1}`;
        const quizData = {
          moduleId,
          moduleName: `Module ${index + 1}`,
          passingScore: 70,
          questions: [
            {
              id: 'q1',
              type: 'multiple-choice',
              question: 'Test',
              points: 100,
              difficulty: 'easy',
              correctAnswer: 0,
              options: ['A', 'B', 'C', 'D']
            }
          ]
        };

        assessmentState.initAssessment(moduleId, quizData);
        assessmentState.saveAnswer('q1', 0);

        const gradedResults = [{
          earnedPoints: score,
          totalPoints: 100,
          isCorrect: score === 100
        }];

        assessmentState.completeAssessment({
          score,
          passed: score >= 70,
          gradedResults
        });
      });

      // Evaluate readiness
      const evaluation = evaluateReadiness();

      expect(evaluation).toBeDefined();
      expect(evaluation.overallScore).toBeGreaterThan(0);
      expect(evaluation.proficiencyLevel).toHaveProperty('label');
      expect(evaluation.skillAreas).toBeDefined();

      // With an average score of ~85.5, should be Expert level
      expect(evaluation.proficiencyLevel.label).toBe('Expert');
      expect(evaluation.readyForProduction).toBe(true);
    });
  });

  describe('Achievement System Integration', () => {
    it('should unlock progressive achievements', () => {
      // Complete first module
      const quizData1 = {
        moduleId: 'module1',
        moduleName: 'Module 1',
        passingScore: 70,
        questions: [{ id: 'q1', type: 'multiple-choice', points: 100, correctAnswer: 0, options: ['A'] }]
      };

      assessmentState.initAssessment('module1', quizData1);
      assessmentState.saveAnswer('q1', 0);
      assessmentState.completeAssessment({ score: 100, passed: true, gradedResults: [] });

      let achievements = checkAchievements({ moduleId: 'module1', score: 100, passed: true });
      const firstStepsAchievement = achievements.find(a => a.id === 'first_steps');
      expect(firstStepsAchievement).toBeDefined();

      // Complete more modules
      ['module2', 'module3', 'module4'].forEach(moduleId => {
        const quizData = {
          moduleId,
          moduleName: moduleId,
          passingScore: 70,
          questions: [{ id: 'q1', type: 'multiple-choice', points: 100, correctAnswer: 0, options: ['A'] }]
        };

        assessmentState.initAssessment(moduleId, quizData);
        assessmentState.saveAnswer('q1', 0);
        assessmentState.completeAssessment({ score: 100, passed: true, gradedResults: [] });

        achievements = checkAchievements({ moduleId, score: 100, passed: true });
      });

      // Complete all 6 modules
      ['module5', 'module6'].forEach(moduleId => {
        const quizData = {
          moduleId,
          moduleName: moduleId,
          passingScore: 70,
          questions: [{ id: 'q1', type: 'multiple-choice', points: 100, correctAnswer: 0, options: ['A'] }]
        };

        assessmentState.initAssessment(moduleId, quizData);
        assessmentState.saveAnswer('q1', 0);
        assessmentState.completeAssessment({ score: 100, passed: true, gradedResults: [] });

        achievements = checkAchievements({ moduleId, score: 100, passed: true });
      });

      // Check for completionist achievement
      const completionistAchievement = achievements.find(a => a.id === 'completionist');
      expect(completionistAchievement).toBeDefined();
    });
  });
});
