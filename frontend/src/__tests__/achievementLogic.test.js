/**
 * Unit tests for achievementLogic.js
 */

import {
  getAllAchievements,
  getUnlockedAchievements,
  isAchievementUnlocked,
  unlockAchievement,
  checkAchievements,
  getAchievementStats,
  trackHintViewed,
  trackCertificateShared,
  clearAchievements
} from '../lib/achievementLogic';

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

// Mock assessmentState
jest.mock('../lib/assessmentState', () => ({
  hasPassedModule: jest.fn((moduleId) => {
    return moduleId === 'module1' || moduleId === 'module2';
  }),
  getModuleHistory: jest.fn((moduleId) => {
    if (moduleId === 'module1') {
      return [{ result: { score: 100, passed: true }, startTime: Date.now() - 300000, endTime: Date.now() }];
    }
    return [];
  }),
  getBestScore: jest.fn((moduleId) => {
    if (moduleId === 'module1') return 100;
    if (moduleId === 'module2') return 85;
    return 0;
  })
}));

describe('achievementLogic', () => {
  beforeEach(() => {
    localStorage.clear();
  });

  describe('getAllAchievements', () => {
    it('should return all available achievements', () => {
      const achievements = getAllAchievements();
      expect(Array.isArray(achievements)).toBe(true);
      expect(achievements.length).toBeGreaterThan(0);
      expect(achievements[0]).toHaveProperty('id');
      expect(achievements[0]).toHaveProperty('name');
      expect(achievements[0]).toHaveProperty('tier');
      expect(achievements[0]).toHaveProperty('points');
    });
  });

  describe('unlockAchievement', () => {
    it('should unlock a new achievement', () => {
      const result = unlockAchievement('first_steps');

      expect(result).toBeTruthy();
      expect(result.id).toBe('first_steps');
      expect(result.unlockedAt).toBeDefined();
    });

    it('should not unlock already unlocked achievement', () => {
      unlockAchievement('first_steps');
      const result = unlockAchievement('first_steps');

      expect(result).toBeNull();
    });

    it('should persist unlocked achievements', () => {
      unlockAchievement('first_steps');
      unlockAchievement('completionist');

      const unlocked = getUnlockedAchievements();
      expect(unlocked.length).toBe(2);
      expect(unlocked.map(a => a.id)).toContain('first_steps');
      expect(unlocked.map(a => a.id)).toContain('completionist');
    });
  });

  describe('isAchievementUnlocked', () => {
    it('should return false for locked achievement', () => {
      expect(isAchievementUnlocked('first_steps')).toBe(false);
    });

    it('should return true for unlocked achievement', () => {
      unlockAchievement('first_steps');
      expect(isAchievementUnlocked('first_steps')).toBe(true);
    });
  });

  describe('checkAchievements', () => {
    it('should unlock module completion achievement', () => {
      const context = {
        moduleId: 'module1',
        passed: true
      };

      const unlocked = checkAchievements(context);

      expect(unlocked.length).toBeGreaterThan(0);
      expect(unlocked.some(a => a.condition?.type === 'module_complete')).toBe(true);
    });

    it('should unlock perfect score achievement', () => {
      const context = {
        moduleId: 'module1',
        score: 100,
        passed: true
      };

      const unlocked = checkAchievements(context);

      const perfectAchievement = unlocked.find(a =>
        a.condition?.type === 'perfect_score'
      );
      expect(perfectAchievement).toBeDefined();
    });

    it('should unlock speed completion achievement', () => {
      const context = {
        moduleId: 'module1',
        passed: true,
        timeSpent: 250000 // 4 minutes 10 seconds
      };

      const unlocked = checkAchievements(context);

      const speedAchievement = unlocked.find(a =>
        a.condition?.type === 'speed_completion' && a.condition?.maxTime === 300000
      );
      expect(speedAchievement).toBeDefined();
    });

    it('should not unlock achievements that do not meet conditions', () => {
      const context = {
        moduleId: 'module1',
        score: 75, // Not perfect
        passed: true,
        timeSpent: 400000 // Too slow
      };

      const unlocked = checkAchievements(context);

      const perfectAchievement = unlocked.find(a =>
        a.condition?.type === 'perfect_score'
      );
      expect(perfectAchievement).toBeUndefined();
    });

    it('should not re-unlock already unlocked achievements', () => {
      const context = {
        moduleId: 'module1',
        passed: true
      };

      // First unlock
      const firstUnlock = checkAchievements(context);
      expect(firstUnlock.length).toBeGreaterThan(0);

      // Try again
      const secondUnlock = checkAchievements(context);
      expect(secondUnlock.length).toBe(0);
    });
  });

  describe('getAchievementStats', () => {
    it('should return correct stats with no achievements', () => {
      const stats = getAchievementStats();

      expect(stats.total).toBeGreaterThan(0);
      expect(stats.unlocked).toBe(0);
      expect(stats.percentage).toBe(0);
      expect(stats.earnedPoints).toBe(0);
    });

    it('should calculate correct stats with unlocked achievements', () => {
      // Unlock a few achievements
      unlockAchievement('first_steps'); // Bronze - 10 points
      unlockAchievement('completionist'); // Gold - 50 points

      const stats = getAchievementStats();

      expect(stats.unlocked).toBe(2);
      expect(stats.earnedPoints).toBe(60);
      expect(stats.percentage).toBeGreaterThan(0);
    });

    it('should break down stats by tier', () => {
      unlockAchievement('first_steps'); // Bronze
      unlockAchievement('completionist'); // Gold

      const stats = getAchievementStats();

      expect(stats.unlockedByTier.bronze).toBe(1);
      expect(stats.unlockedByTier.gold).toBe(1);
    });

    it('should break down stats by category', () => {
      const stats = getAchievementStats();

      expect(stats.byCategory).toBeDefined();
      expect(typeof stats.byCategory).toBe('object');
    });
  });

  describe('trackHintViewed', () => {
    it('should track hint views', () => {
      trackHintViewed('module1', 'q1');
      trackHintViewed('module1', 'q2');
      trackHintViewed('module2', 'q1');

      const data = JSON.parse(localStorage.getItem('jadevectordb_hint_tracking') || '{}');

      expect(data.module1).toContain('q1');
      expect(data.module1).toContain('q2');
      expect(data.module2).toContain('q1');
    });
  });

  describe('trackCertificateShared', () => {
    it('should track certificate shares', () => {
      trackCertificateShared('linkedin');
      trackCertificateShared('twitter');

      const data = JSON.parse(localStorage.getItem('jadevectordb_certificate_shares') || '{}');

      expect(data.linkedin).toBe(1);
      expect(data.twitter).toBe(1);
    });

    it('should increment share count', () => {
      trackCertificateShared('linkedin');
      trackCertificateShared('linkedin');

      const data = JSON.parse(localStorage.getItem('jadevectordb_certificate_shares') || '{}');

      expect(data.linkedin).toBe(2);
    });
  });

  describe('clearAchievements', () => {
    it('should clear all unlocked achievements', () => {
      unlockAchievement('first_steps');
      unlockAchievement('completionist');

      clearAchievements();

      const unlocked = getUnlockedAchievements();
      expect(unlocked.length).toBe(0);
    });
  });
});
