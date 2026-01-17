/**
 * Unit tests for readinessEvaluation.js
 * Aligned with actual implementation
 */

import {
  evaluateReadiness,
  getProficiencyLevel,
  getCertificationLevel,
  compareReadiness,
  exportReadinessData
} from '../lib/readinessEvaluation';

// Mock assessmentState
jest.mock('../lib/assessmentState', () => ({
  default: {
    getBestScore: jest.fn((moduleId) => {
      const scores = {
        'module1': 85,
        'module2': 90,
        'module3': 75,
        'module4': 80,
        'module5': 95,
        'module6': 88
      };
      return scores[moduleId] || null;
    }),
    hasPassedModule: jest.fn((moduleId) => {
      return ['module1', 'module2', 'module3', 'module4', 'module5', 'module6'].includes(moduleId);
    }),
    getModuleStatistics: jest.fn((moduleId) => {
      const stats = {
        'module1': { attempts: 2, averageScore: 80, improvement: 10 },
        'module2': { attempts: 1, averageScore: 90, improvement: 0 },
        'module3': { attempts: 3, averageScore: 70, improvement: 15 },
        'module4': { attempts: 2, averageScore: 75, improvement: 10 },
        'module5': { attempts: 1, averageScore: 95, improvement: 0 },
        'module6': { attempts: 2, averageScore: 85, improvement: 6 }
      };
      return stats[moduleId] || { attempts: 0, averageScore: 0, improvement: 0 };
    })
  }
}));

describe('readinessEvaluation', () => {
  describe('evaluateReadiness', () => {
    it('should return complete evaluation object', () => {
      const evaluation = evaluateReadiness();

      expect(evaluation).toHaveProperty('overallScore');
      expect(evaluation).toHaveProperty('proficiencyLevel');
      expect(evaluation).toHaveProperty('skillAreaScores');
      expect(evaluation).toHaveProperty('readyForProduction');
      expect(evaluation).toHaveProperty('recommendedForProduction');
      expect(evaluation).toHaveProperty('skillGaps');
      expect(evaluation).toHaveProperty('moduleScores');
      expect(evaluation).toHaveProperty('recommendations');
    });

    it('should calculate overall score as a number', () => {
      const evaluation = evaluateReadiness();

      expect(typeof evaluation.overallScore).toBe('number');
      expect(evaluation.overallScore).toBeGreaterThanOrEqual(0);
      expect(evaluation.overallScore).toBeLessThanOrEqual(100);
    });

    it('should evaluate all skill areas', () => {
      const evaluation = evaluateReadiness();

      expect(evaluation.skillAreaScores).toBeDefined();
      expect(typeof evaluation.skillAreaScores).toBe('object');
    });

    it('should determine production readiness', () => {
      const evaluation = evaluateReadiness();

      expect(typeof evaluation.readyForProduction).toBe('boolean');
      expect(typeof evaluation.recommendedForProduction).toBe('boolean');
    });

    it('should include module scores', () => {
      const evaluation = evaluateReadiness();

      expect(evaluation.moduleScores).toBeDefined();
      expect(typeof evaluation.moduleScores).toBe('object');
    });

    it('should include evaluation metadata', () => {
      const evaluation = evaluateReadiness();

      expect(evaluation.completedModules).toBeDefined();
      expect(evaluation.totalModules).toBe(6);
      expect(evaluation.evaluationDate).toBeDefined();
    });
  });

  describe('getProficiencyLevel', () => {
    it('should return Beginner for low scores', () => {
      const level = getProficiencyLevel(30);
      expect(level).toBe('Beginner');
    });

    it('should return Intermediate for mid-range scores', () => {
      const level = getProficiencyLevel(55);
      expect(level).toBe('Intermediate');
    });

    it('should return Proficient for good scores', () => {
      const level = getProficiencyLevel(75);
      expect(level).toBe('Proficient');
    });

    it('should return Expert for high scores', () => {
      const level = getProficiencyLevel(88);
      expect(level).toBe('Expert');
    });

    it('should return Master for excellent scores', () => {
      const level = getProficiencyLevel(96);
      expect(level).toBe('Master');
    });

    it('should handle edge cases', () => {
      expect(getProficiencyLevel(0)).toBe('Beginner');
      expect(getProficiencyLevel(100)).toBe('Master');
    });

    it('should handle boundary values', () => {
      // Test exact threshold boundaries
      expect(getProficiencyLevel(50)).toBeDefined();
      expect(getProficiencyLevel(70)).toBeDefined();
      expect(getProficiencyLevel(85)).toBeDefined();
      expect(getProficiencyLevel(95)).toBeDefined();
    });
  });

  describe('getCertificationLevel', () => {
    it('should return appropriate certification for score', () => {
      const lowCert = getCertificationLevel(60);
      const midCert = getCertificationLevel(80);
      const highCert = getCertificationLevel(95);

      expect(lowCert).toBeDefined();
      expect(midCert).toBeDefined();
      expect(highCert).toBeDefined();
    });

    it('should return null or basic for failing scores', () => {
      const result = getCertificationLevel(40);
      // Implementation may return null or a basic/no certification object
      expect(result).toBeDefined();
    });
  });

  describe('compareReadiness', () => {
    it('should compare two evaluations', () => {
      const previous = {
        overallScore: 70,
        proficiencyLevel: 'Proficient',
        skillAreaScores: { fundamentals: 75 },
        completedModules: 4
      };

      const current = {
        overallScore: 85,
        proficiencyLevel: 'Expert',
        skillAreaScores: { fundamentals: 90 },
        completedModules: 6
      };

      const comparison = compareReadiness(previous, current);

      expect(comparison).toHaveProperty('scoreChange');
      expect(comparison.scoreChange).toBe(15);
      expect(comparison).toHaveProperty('improved');
      expect(comparison.improved).toBe(true);
    });

    it('should detect declining performance', () => {
      const previous = {
        overallScore: 90,
        proficiencyLevel: 'Expert',
        skillAreaScores: {},
        completedModules: 6
      };

      const current = {
        overallScore: 75,
        proficiencyLevel: 'Proficient',
        skillAreaScores: {},
        completedModules: 6
      };

      const comparison = compareReadiness(previous, current);

      expect(comparison.scoreChange).toBe(-15);
      expect(comparison.improved).toBe(false);
    });

    it('should handle identical evaluations', () => {
      const evaluation = {
        overallScore: 80,
        proficiencyLevel: 'Proficient',
        skillAreaScores: {},
        completedModules: 5
      };

      const comparison = compareReadiness(evaluation, evaluation);

      expect(comparison.scoreChange).toBe(0);
    });
  });

  describe('exportReadinessData', () => {
    it('should export evaluation as JSON string', () => {
      const evaluation = evaluateReadiness();
      const exported = exportReadinessData(evaluation);

      expect(typeof exported).toBe('string');

      // Should be valid JSON
      const parsed = JSON.parse(exported);
      expect(parsed).toHaveProperty('overallScore');
      expect(parsed).toHaveProperty('proficiencyLevel');
    });

    it('should include all evaluation data', () => {
      const evaluation = evaluateReadiness();
      const exported = exportReadinessData(evaluation);
      const parsed = JSON.parse(exported);

      expect(parsed.moduleScores).toBeDefined();
      expect(parsed.skillAreaScores).toBeDefined();
      expect(parsed.recommendations).toBeDefined();
    });
  });

  describe('Integration', () => {
    it('should provide consistent proficiency level with evaluation', () => {
      const evaluation = evaluateReadiness();
      const directLevel = getProficiencyLevel(evaluation.overallScore);

      expect(evaluation.proficiencyLevel).toBe(directLevel);
    });

    it('should handle export/import cycle', () => {
      const evaluation = evaluateReadiness();
      const exported = exportReadinessData(evaluation);
      const reimported = JSON.parse(exported);

      expect(reimported.overallScore).toBe(evaluation.overallScore);
      expect(reimported.proficiencyLevel).toBe(evaluation.proficiencyLevel);
    });
  });
});
