/**
 * Achievement Logic
 *
 * Handles achievement checking, unlocking, and tracking
 */

import achievementsData from '../data/achievements.json';
import assessmentState from './assessmentState';

const STORAGE_KEY = 'jadevectordb_achievements';

/**
 * Get all achievement definitions
 */
export function getAllAchievements() {
  return achievementsData.achievements;
}

/**
 * Get user's unlocked achievements
 */
export function getUnlockedAchievements() {
  try {
    const data = localStorage.getItem(STORAGE_KEY);
    if (!data) return [];
    const parsed = JSON.parse(data);
    return parsed.unlocked || [];
  } catch (error) {
    console.error('Failed to load achievements:', error);
    return [];
  }
}

/**
 * Check if achievement is unlocked
 */
export function isAchievementUnlocked(achievementId) {
  const unlocked = getUnlockedAchievements();
  return unlocked.some(a => a.id === achievementId);
}

/**
 * Unlock an achievement
 */
export function unlockAchievement(achievementId) {
  if (isAchievementUnlocked(achievementId)) {
    return null; // Already unlocked
  }

  const achievement = achievementsData.achievements.find(a => a.id === achievementId);
  if (!achievement) {
    console.error(`Achievement not found: ${achievementId}`);
    return null;
  }

  const unlocked = getUnlockedAchievements();
  const newAchievement = {
    ...achievement,
    unlockedAt: new Date().toISOString(),
    timestamp: Date.now()
  };

  unlocked.push(newAchievement);

  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify({ unlocked }));
    return newAchievement;
  } catch (error) {
    console.error('Failed to save achievement:', error);
    return null;
  }
}

/**
 * Check all achievements and unlock any that are earned
 */
export function checkAchievements(context = {}) {
  const newlyUnlocked = [];
  const allAchievements = getAllAchievements();

  for (const achievement of allAchievements) {
    if (isAchievementUnlocked(achievement.id)) {
      continue; // Skip already unlocked
    }

    if (checkCondition(achievement.condition, context)) {
      const unlocked = unlockAchievement(achievement.id);
      if (unlocked) {
        newlyUnlocked.push(unlocked);
      }
    }
  }

  return newlyUnlocked;
}

/**
 * Check if achievement condition is met
 */
function checkCondition(condition, context) {
  switch (condition.type) {
    case 'module_complete':
      return checkModuleComplete(condition.module);

    case 'perfect_score':
      return checkPerfectScore(condition.module);

    case 'speed_completion':
      return checkSpeedCompletion(condition.maxTime, context);

    case 'first_attempt_pass':
      return checkFirstAttemptPass(context);

    case 'retry_success':
      return checkRetrySuccess(condition.minAttempts, context);

    case 'all_modules_complete':
      return checkAllModulesComplete(condition.count);

    case 'all_first_attempt':
      return checkAllFirstAttempt(condition.count);

    case 'all_perfect_scores':
      return checkAllPerfectScores(condition.count);

    case 'readiness_level':
      return checkReadinessLevel(condition.minLevel || condition.level, context);

    case 'time_of_day':
      return checkTimeOfDay(condition);

    case 'hints_viewed':
      return checkHintsViewed(condition.count);

    case 'score_improvement':
      return checkScoreImprovement(condition.minImprovement, context);

    case 'daily_streak':
      return checkDailyStreak(condition.days);

    case 'same_day_completion':
      return checkSameDayCompletion(condition.count);

    case 'certificate_shared':
      return checkCertificateShared();

    default:
      console.warn(`Unknown condition type: ${condition.type}`);
      return false;
  }
}

// Condition checking functions

function checkModuleComplete(moduleId) {
  return assessmentState.hasPassedModule(moduleId);
}

function checkPerfectScore(moduleId) {
  const bestScore = assessmentState.getBestScore(moduleId);
  return bestScore === 100;
}

function checkSpeedCompletion(maxTime, context) {
  if (!context.timeSpent) return false;
  return context.timeSpent <= maxTime;
}

function checkFirstAttemptPass(context) {
  if (!context.moduleId) return false;
  const history = assessmentState.getModuleHistory(context.moduleId);
  return history.length === 1 && history[0].passed;
}

function checkRetrySuccess(minAttempts, context) {
  if (!context.moduleId) return false;
  const history = assessmentState.getModuleHistory(context.moduleId);
  return history.length >= minAttempts && history[history.length - 1].passed;
}

function checkAllModulesComplete(count) {
  const modules = ['module1', 'module2', 'module3', 'module4', 'module5', 'module6'];
  const completed = modules.filter(m => assessmentState.hasPassedModule(m));
  return completed.length >= count;
}

function checkAllFirstAttempt(count) {
  const modules = ['module1', 'module2', 'module3', 'module4', 'module5', 'module6'];
  let firstAttemptPasses = 0;

  for (const moduleId of modules) {
    const history = assessmentState.getModuleHistory(moduleId);
    if (history.length === 1 && history[0].passed) {
      firstAttemptPasses++;
    }
  }

  return firstAttemptPasses >= count;
}

function checkAllPerfectScores(count) {
  const modules = ['module1', 'module2', 'module3', 'module4', 'module5', 'module6'];
  let perfectScores = 0;

  for (const moduleId of modules) {
    const bestScore = assessmentState.getBestScore(moduleId);
    if (bestScore === 100) {
      perfectScores++;
    }
  }

  return perfectScores >= count;
}

function checkReadinessLevel(level, context) {
  if (!context.readinessLevel) return false;

  const levels = ['beginner', 'intermediate', 'proficient', 'expert', 'master'];
  const requiredIndex = levels.indexOf(level);
  const actualIndex = levels.indexOf(context.readinessLevel.toLowerCase());

  return actualIndex >= requiredIndex;
}

function checkTimeOfDay(condition) {
  const hour = new Date().getHours();
  if (condition.maxHour !== undefined && condition.minHour !== undefined) {
    return hour >= condition.minHour && hour < condition.maxHour;
  }
  if (condition.maxHour !== undefined) {
    return hour < condition.maxHour;
  }
  if (condition.minHour !== undefined) {
    return hour >= condition.minHour;
  }
  return false;
}

function checkHintsViewed(count) {
  // This would need to be tracked separately
  // For now, return false as we don't track this yet
  const hintsData = localStorage.getItem('jadevectordb_hints_viewed');
  if (!hintsData) return false;

  try {
    const hints = JSON.parse(hintsData);
    return hints.length >= count;
  } catch {
    return false;
  }
}

function checkScoreImprovement(minImprovement, context) {
  if (!context.moduleId) return false;
  const history = assessmentState.getModuleHistory(context.moduleId);
  if (history.length < 2) return false;

  const firstScore = history[0].score;
  const latestScore = history[history.length - 1].score;
  const improvement = latestScore - firstScore;

  return improvement >= minImprovement;
}

function checkDailyStreak(days) {
  // This would need to track completion dates
  // Simplified implementation
  const modules = ['module1', 'module2', 'module3', 'module4', 'module5', 'module6'];
  const completionDates = [];

  for (const moduleId of modules) {
    const history = assessmentState.getModuleHistory(moduleId);
    if (history.length > 0) {
      completionDates.push(new Date(history[history.length - 1].date));
    }
  }

  if (completionDates.length < days) return false;

  // Sort dates
  completionDates.sort((a, b) => a - b);

  // Check for consecutive days
  let streak = 1;
  for (let i = 1; i < completionDates.length; i++) {
    const dayDiff = Math.floor((completionDates[i] - completionDates[i - 1]) / (1000 * 60 * 60 * 24));
    if (dayDiff === 1) {
      streak++;
      if (streak >= days) return true;
    } else if (dayDiff > 1) {
      streak = 1;
    }
  }

  return streak >= days;
}

function checkSameDayCompletion(count) {
  const modules = ['module1', 'module2', 'module3', 'module4', 'module5', 'module6'];
  const completionDates = [];

  for (const moduleId of modules) {
    const history = assessmentState.getModuleHistory(moduleId);
    if (history.length > 0) {
      const date = new Date(history[history.length - 1].date);
      completionDates.push(date.toDateString());
    }
  }

  // Count occurrences of each date
  const dateCounts = {};
  completionDates.forEach(date => {
    dateCounts[date] = (dateCounts[date] || 0) + 1;
  });

  // Check if any date has enough completions
  return Object.values(dateCounts).some(count => count >= count);
}

function checkCertificateShared() {
  // Check if certificate share was tracked
  const shared = localStorage.getItem('jadevectordb_certificate_shared');
  return shared === 'true';
}

/**
 * Get achievement stats
 */
export function getAchievementStats() {
  const all = getAllAchievements();
  const unlocked = getUnlockedAchievements();

  const totalPoints = all.reduce((sum, a) => sum + a.points, 0);
  const earnedPoints = unlocked.reduce((sum, a) => sum + a.points, 0);

  const byTier = {};
  const byCategory = {};

  all.forEach(achievement => {
    byTier[achievement.tier] = (byTier[achievement.tier] || 0) + 1;
    byCategory[achievement.category] = (byCategory[achievement.category] || 0) + 1;
  });

  const unlockedByTier = {};
  const unlockedByCategory = {};

  unlocked.forEach(achievement => {
    unlockedByTier[achievement.tier] = (unlockedByTier[achievement.tier] || 0) + 1;
    unlockedByCategory[achievement.category] = (unlockedByCategory[achievement.category] || 0) + 1;
  });

  return {
    total: all.length,
    unlocked: unlocked.length,
    percentage: Math.round((unlocked.length / all.length) * 100),
    totalPoints,
    earnedPoints,
    byTier,
    byCategory,
    unlockedByTier,
    unlockedByCategory
  };
}

/**
 * Track hint viewed (helper for achievements)
 */
export function trackHintViewed(questionId) {
  try {
    const hintsData = localStorage.getItem('jadevectordb_hints_viewed');
    const hints = hintsData ? JSON.parse(hintsData) : [];

    if (!hints.includes(questionId)) {
      hints.push(questionId);
      localStorage.setItem('jadevectordb_hints_viewed', JSON.stringify(hints));
    }
  } catch (error) {
    console.error('Failed to track hint:', error);
  }
}

/**
 * Track certificate shared (helper for achievements)
 */
export function trackCertificateShared() {
  localStorage.setItem('jadevectordb_certificate_shared', 'true');
}

/**
 * Clear all achievements (for testing/reset)
 */
export function clearAchievements() {
  localStorage.removeItem(STORAGE_KEY);
  localStorage.removeItem('jadevectordb_hints_viewed');
  localStorage.removeItem('jadevectordb_certificate_shared');
}

export default {
  getAllAchievements,
  getUnlockedAchievements,
  isAchievementUnlocked,
  unlockAchievement,
  checkAchievements,
  getAchievementStats,
  trackHintViewed,
  trackCertificateShared,
  clearAchievements
};
