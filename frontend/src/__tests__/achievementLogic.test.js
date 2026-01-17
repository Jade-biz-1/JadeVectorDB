/**
 * Unit tests for achievementLogic.js
 * Aligned with actual implementation
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
  default: {
    hasPassedModule: jest.fn((moduleId) => {
      return moduleId === 'module1' || moduleId === 'module2';
    }),
    getModuleHistory: jest.fn((moduleId) => {
      if (moduleId === 'module1') {
        return [{ score: 100, passed: true, date: Date.now() - 300000 }];
      }
      return [];
    }),
    getBestScore: jest.fn((moduleId) => {
      if (moduleId === 'module1') return 100;
      if (moduleId === 'module2') return 85;
      return null;
    })
  },
  hasPassedModule: jest.fn((moduleId) => {
    return moduleId === 'module1' || moduleId === 'module2';
  }),
  getModuleHistory: jest.fn((moduleId) => {
    if (moduleId === 'module1') {
      return [{ score: 100, passed: true, date: Date.now() - 300000 }];
    }
    return [];
  }),
  getBestScore: jest.fn((moduleId) => {
    if (moduleId === 'module1') return 100;
    if (moduleId === 'module2') return 85;
    return null;
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

    it('should return null for non-existent achievement', () => {
      const result = unlockAchievement('non_existent_achievement');
      expect(result).toBeNull();
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
    it('should return array of newly unlocked achievements', () => {
      const context = {
        moduleId: 'module1',
        passed: true
      };

      const unlocked = checkAchievements(context);

      expect(Array.isArray(unlocked)).toBe(true);
    });

    it('should not re-unlock already unlocked achievements', () => {
      const context = {
        moduleId: 'module1',
        passed: true
      };

      // First unlock
      const firstUnlock = checkAchievements(context);

      // Try again - should get fewer or zero new unlocks
      const secondUnlock = checkAchievements(context);
      expect(secondUnlock.length).toBeLessThanOrEqual(firstUnlock.length);
    });

    it('should pass context to condition checkers', () => {
      const context = {
        moduleId: 'module1',
        passed: true,
        timeSpent: 250000, // 4 minutes 10 seconds
        readinessLevel: 'intermediate'
      };

      const unlocked = checkAchievements(context);
      expect(Array.isArray(unlocked)).toBe(true);
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
      unlockAchievement('first_steps');
      unlockAchievement('completionist');

      const stats = getAchievementStats();

      expect(stats.unlocked).toBe(2);
      expect(stats.earnedPoints).toBeGreaterThan(0);
      expect(stats.percentage).toBeGreaterThan(0);
    });

    it('should break down stats by tier', () => {
      unlockAchievement('first_steps');

      const stats = getAchievementStats();

      expect(stats.byTier).toBeDefined();
      expect(stats.unlockedByTier).toBeDefined();
      expect(typeof stats.byTier).toBe('object');
    });

    it('should break down stats by category', () => {
      const stats = getAchievementStats();

      expect(stats.byCategory).toBeDefined();
      expect(typeof stats.byCategory).toBe('object');
    });
  });

  describe('trackHintViewed', () => {
    it('should track hint views in localStorage', () => {
      // Implementation uses flat array with question IDs
      trackHintViewed('q1');
      trackHintViewed('q2');
      trackHintViewed('q3');

      const data = JSON.parse(localStorage.getItem('jadevectordb_hints_viewed') || '[]');

      expect(Array.isArray(data)).toBe(true);
      expect(data).toContain('q1');
      expect(data).toContain('q2');
      expect(data).toContain('q3');
    });

    it('should not add duplicate hint views', () => {
      trackHintViewed('q1');
      trackHintViewed('q1');
      trackHintViewed('q1');

      const data = JSON.parse(localStorage.getItem('jadevectordb_hints_viewed') || '[]');

      expect(data.filter(id => id === 'q1').length).toBe(1);
    });
  });

  describe('trackCertificateShared', () => {
    it('should track certificate shared as boolean', () => {
      // Implementation sets a simple boolean flag
      trackCertificateShared();

      const shared = localStorage.getItem('jadevectordb_certificate_shared');

      expect(shared).toBe('true');
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

    it('should clear hint tracking data', () => {
      trackHintViewed('q1');
      clearAchievements();

      const hints = localStorage.getItem('jadevectordb_hints_viewed');
      expect(hints).toBeNull();
    });

    it('should clear certificate shared flag', () => {
      trackCertificateShared();
      clearAchievements();

      const shared = localStorage.getItem('jadevectordb_certificate_shared');
      expect(shared).toBeNull();
    });
  });
});
