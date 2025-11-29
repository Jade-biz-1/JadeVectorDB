/**
 * Unit tests for readinessEvaluation.js
 */

import {
  evaluateReadiness,
  getProficiencyLevel,
  getSkillGaps,
  getRecommendations
} from '../lib/readinessEvaluation';

// Mock assessmentState
jest.mock('../lib/assessmentState', () => ({
  getBestScore: jest.fn((moduleId) => {
    const scores = {
      'module1': 85,
      'module2': 90,
      'module3': 75,
      'module4': 80,
      'module5': 95,
      'module6': 88
    };
    return scores[moduleId] || 0;
  }),
  hasPassedModule: jest.fn((moduleId) => {
    return ['module1', 'module2', 'module3', 'module4', 'module5', 'module6'].includes(moduleId);
  })
}));

describe('readinessEvaluation', () => {
  describe('evaluateReadiness', () => {
    it('should return complete evaluation object', () => {
      const evaluation = evaluateReadiness();

      expect(evaluation).toHaveProperty('overallScore');
      expect(evaluation).toHaveProperty('proficiencyLevel');
      expect(evaluation).toHaveProperty('skillAreas');
      expect(evaluation).toHaveProperty('readyForProduction');
      expect(evaluation).toHaveProperty('recommendedForProduction');
      expect(evaluation).toHaveProperty('skillGaps');
      expect(evaluation).toHaveProperty('moduleScores');
    });

    it('should calculate correct overall score', () => {
      const evaluation = evaluateReadiness();

      // Based on mocked scores: 85, 90, 75, 80, 95, 88
      // Average = (85 + 90 + 75 + 80 + 95 + 88) / 6 = 85.5
      expect(evaluation.overallScore).toBeCloseTo(85.5, 0);
    });

    it('should evaluate all skill areas', () => {
      const evaluation = evaluateReadiness();

      expect(evaluation.skillAreas).toHaveProperty('fundamentals');
      expect(evaluation.skillAreas).toHaveProperty('vectorSearch');
      expect(evaluation.skillAreas).toHaveProperty('optimization');
      expect(evaluation.skillAreas).toHaveProperty('production');

      // Each skill area should have score and skills
      expect(evaluation.skillAreas.fundamentals).toHaveProperty('score');
      expect(evaluation.skillAreas.fundamentals).toHaveProperty('skills');
    });

    it('should determine production readiness correctly', () => {
      const evaluation = evaluateReadiness();

      // With overall score of 85.5, should be ready
      expect(evaluation.readyForProduction).toBe(true);
      expect(evaluation.recommendedForProduction).toBe(true);
    });

    it('should include module scores', () => {
      const evaluation = evaluateReadiness();

      expect(evaluation.moduleScores).toHaveProperty('module1');
      expect(evaluation.moduleScores).toHaveProperty('module2');
      expect(evaluation.moduleScores.module1).toBe(85);
      expect(evaluation.moduleScores.module2).toBe(90);
    });
  });

  describe('getProficiencyLevel', () => {
    it('should return Beginner for scores 0-39', () => {
      expect(getProficiencyLevel(20).label).toBe('Beginner');
      expect(getProficiencyLevel(39).label).toBe('Beginner');
    });

    it('should return Intermediate for scores 40-59', () => {
      expect(getProficiencyLevel(40).label).toBe('Intermediate');
      expect(getProficiencyLevel(59).label).toBe('Intermediate');
    });

    it('should return Proficient for scores 60-74', () => {
      expect(getProficiencyLevel(60).label).toBe('Proficient');
      expect(getProficiencyLevel(74).label).toBe('Proficient');
    });

    it('should return Expert for scores 75-89', () => {
      expect(getProficiencyLevel(75).label).toBe('Expert');
      expect(getProficiencyLevel(89).label).toBe('Expert');
    });

    it('should return Master for scores 90-100', () => {
      expect(getProficiencyLevel(90).label).toBe('Master');
      expect(getProficiencyLevel(100).label).toBe('Master');
    });

    it('should include description and color', () => {
      const level = getProficiencyLevel(85);

      expect(level).toHaveProperty('label');
      expect(level).toHaveProperty('description');
      expect(level).toHaveProperty('color');
      expect(level).toHaveProperty('min');
      expect(level).toHaveProperty('max');
    });
  });

  describe('getSkillGaps', () => {
    it('should identify gaps for low-performing areas', () => {
      const skillAreas = {
        fundamentals: {
          score: 85,
          skills: [
            { name: 'Basic Concepts', mastered: true },
            { name: 'Vector Storage', mastered: true }
          ]
        },
        vectorSearch: {
          score: 60,
          skills: [
            { name: 'Search Algorithms', mastered: false },
            { name: 'Query Optimization', mastered: false }
          ]
        },
        optimization: {
          score: 55,
          skills: [
            { name: 'Index Selection', mastered: false }
          ]
        },
        production: {
          score: 90,
          skills: [
            { name: 'Deployment', mastered: true }
          ]
        }
      };

      const gaps = getSkillGaps(skillAreas);

      expect(gaps.length).toBeGreaterThan(0);

      // Should identify vectorSearch and optimization as gaps
      const gapAreas = gaps.map(g => g.area);
      expect(gapAreas).toContain('vectorSearch');
      expect(gapAreas).toContain('optimization');
      expect(gapAreas).not.toContain('fundamentals');
      expect(gapAreas).not.toContain('production');
    });

    it('should return empty array when all areas are strong', () => {
      const skillAreas = {
        fundamentals: { score: 90, skills: [] },
        vectorSearch: { score: 85, skills: [] },
        optimization: { score: 88, skills: [] },
        production: { score: 92, skills: [] }
      };

      const gaps = getSkillGaps(skillAreas);

      expect(gaps.length).toBe(0);
    });

    it('should list specific unmastered skills', () => {
      const skillAreas = {
        fundamentals: { score: 85, skills: [] },
        vectorSearch: {
          score: 60,
          skills: [
            { name: 'Search Algorithms', mastered: false },
            { name: 'Query Optimization', mastered: false },
            { name: 'Basic Search', mastered: true }
          ]
        },
        optimization: { score: 85, skills: [] },
        production: { score: 85, skills: [] }
      };

      const gaps = getSkillGaps(skillAreas);

      expect(gaps.length).toBe(1);
      expect(gaps[0].area).toBe('vectorSearch');
      expect(gaps[0].unmasteredSkills).toEqual(['Search Algorithms', 'Query Optimization']);
    });
  });

  describe('getRecommendations', () => {
    it('should return recommendations for Beginner level', () => {
      const recommendations = getRecommendations('Beginner', []);

      expect(recommendations).toHaveProperty('nextSteps');
      expect(recommendations).toHaveProperty('practiceProjects');
      expect(recommendations).toHaveProperty('resources');
      expect(Array.isArray(recommendations.nextSteps)).toBe(true);
    });

    it('should return recommendations for Expert level', () => {
      const recommendations = getRecommendations('Expert', []);

      expect(recommendations).toHaveProperty('nextSteps');
      expect(recommendations.nextSteps.length).toBeGreaterThan(0);
    });

    it('should include targeted recommendations for skill gaps', () => {
      const skillGaps = [
        {
          area: 'vectorSearch',
          score: 60,
          unmasteredSkills: ['Search Algorithms']
        }
      ];

      const recommendations = getRecommendations('Proficient', skillGaps);

      expect(recommendations).toHaveProperty('focusAreas');
      expect(recommendations.focusAreas).toBeDefined();
    });

    it('should handle Master level with no gaps', () => {
      const recommendations = getRecommendations('Master', []);

      expect(recommendations).toBeDefined();
      expect(recommendations.nextSteps).toBeDefined();
    });

    it('should suggest practice projects appropriate to level', () => {
      const beginnerRecs = getRecommendations('Beginner', []);
      const expertRecs = getRecommendations('Expert', []);

      expect(beginnerRecs.practiceProjects).toBeDefined();
      expect(expertRecs.practiceProjects).toBeDefined();

      // Practice projects should be arrays
      expect(Array.isArray(beginnerRecs.practiceProjects)).toBe(true);
      expect(Array.isArray(expertRecs.practiceProjects)).toBe(true);
    });
  });

  describe('Edge Cases', () => {
    it('should handle evaluation with no completed modules', () => {
      // Mock assessmentState to return no completed modules
      const assessmentState = require('../lib/assessmentState');
      assessmentState.hasPassedModule.mockReturnValue(false);
      assessmentState.getBestScore.mockReturnValue(0);

      const evaluation = evaluateReadiness();

      expect(evaluation.overallScore).toBe(0);
      expect(evaluation.proficiencyLevel.label).toBe('Beginner');
      expect(evaluation.readyForProduction).toBe(false);
    });

    it('should handle perfect scores across all modules', () => {
      const assessmentState = require('../lib/assessmentState');
      assessmentState.hasPassedModule.mockReturnValue(true);
      assessmentState.getBestScore.mockReturnValue(100);

      const evaluation = evaluateReadiness();

      expect(evaluation.overallScore).toBe(100);
      expect(evaluation.proficiencyLevel.label).toBe('Master');
      expect(evaluation.readyForProduction).toBe(true);
      expect(evaluation.recommendedForProduction).toBe(true);
    });
  });
});
