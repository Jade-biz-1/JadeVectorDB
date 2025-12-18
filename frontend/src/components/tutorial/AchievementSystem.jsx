import React, { useState, useEffect } from 'react';
import {
  getAllAchievements,
  getUnlockedAchievements,
  isAchievementUnlocked,
  getAchievementStats
} from '../../lib/achievementLogic';
import Badge from './Badge';

/**
 * AchievementSystem - Main achievement display and tracking system
 */
const AchievementSystem = ({ onClose }) => {
  const [allAchievements, setAllAchievements] = useState([]);
  const [unlockedAchievements, setUnlockedAchievements] = useState([]);
  const [stats, setStats] = useState(null);
  const [filter, setFilter] = useState('all'); // all, unlocked, locked
  const [categoryFilter, setCategoryFilter] = useState('all');
  const [tierFilter, setTierFilter] = useState('all');

  // Load achievements on mount
  useEffect(() => {
    loadAchievements();
  }, []);

  const loadAchievements = () => {
    const all = getAllAchievements();
    const unlocked = getUnlockedAchievements();
    const achievementStats = getAchievementStats();

    setAllAchievements(all);
    setUnlockedAchievements(unlocked);
    setStats(achievementStats);
  };

  // Get unique categories
  const categories = ['all', ...new Set(allAchievements.map(a => a.category))];
  const tiers = ['all', 'bronze', 'silver', 'gold', 'platinum'];

  // Filter achievements
  const filteredAchievements = allAchievements.filter(achievement => {
    // Status filter
    if (filter === 'unlocked' && !isAchievementUnlocked(achievement.id)) return false;
    if (filter === 'locked' && isAchievementUnlocked(achievement.id)) return false;

    // Category filter
    if (categoryFilter !== 'all' && achievement.category !== categoryFilter) return false;

    // Tier filter
    if (tierFilter !== 'all' && achievement.tier !== tierFilter) return false;

    return true;
  });

  // Get unlock date for achievement
  const getUnlockDate = (achievementId) => {
    const unlocked = unlockedAchievements.find(a => a.id === achievementId);
    return unlocked?.unlockedAt;
  };

  if (!stats) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="animate-spin rounded-full h-16 w-16 border-b-4 border-blue-600"></div>
      </div>
    );
  }

  return (
    <div className="max-w-7xl mx-auto p-6 space-y-6">
      {/* Header */}
      <div className="text-center">
        <h1 className="text-4xl font-bold text-gray-800 mb-2">
          🏆 Achievements
        </h1>
        <p className="text-lg text-gray-600">
          Track your progress and unlock badges as you master JadeVectorDB
        </p>
      </div>

      {/* Stats Overview */}
      <div className="bg-white rounded-lg shadow-xl p-6">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
          {/* Total Progress */}
          <div className="text-center">
            <div className="text-4xl font-bold text-blue-600">
              {stats.percentage}%
            </div>
            <div className="text-sm text-gray-600 mt-1">Overall Progress</div>
            <div className="text-xs text-gray-500">
              {stats.unlocked}/{stats.total} unlocked
            </div>
          </div>

          {/* Total Points */}
          <div className="text-center">
            <div className="text-4xl font-bold text-green-600">
              {stats.earnedPoints}
            </div>
            <div className="text-sm text-gray-600 mt-1">Points Earned</div>
            <div className="text-xs text-gray-500">
              of {stats.totalPoints} total
            </div>
          </div>

          {/* By Tier */}
          <div className="text-center">
            <div className="flex justify-center gap-1 mb-2">
              {['bronze', 'silver', 'gold', 'platinum'].map(tier => (
                <div key={tier} className="text-center">
                  <div className="text-lg font-bold">
                    {stats.unlockedByTier[tier] || 0}
                  </div>
                  <div className="text-xs text-gray-500">
                    {tier.charAt(0).toUpperCase()}
                  </div>
                </div>
              ))}
            </div>
            <div className="text-sm text-gray-600">Badges by Tier</div>
          </div>

          {/* Completion Streak */}
          <div className="text-center">
            <div className="text-4xl font-bold text-purple-600">
              {Math.round((stats.earnedPoints / stats.totalPoints) * 100)}%
            </div>
            <div className="text-sm text-gray-600 mt-1">Points Progress</div>
            <div className="text-xs text-gray-500">
              Keep collecting!
            </div>
          </div>
        </div>

        {/* Progress Bar */}
        <div className="mt-6">
          <div className="flex justify-between text-sm text-gray-600 mb-2">
            <span>Achievement Progress</span>
            <span>{stats.unlocked}/{stats.total}</span>
          </div>
          <div className="w-full bg-gray-200 rounded-full h-4">
            <div
              className="bg-gradient-to-r from-blue-500 to-purple-600 h-4 rounded-full transition-all duration-500"
              style={{ width: `${stats.percentage}%` }}
            />
          </div>
        </div>
      </div>

      {/* Filters */}
      <div className="bg-white rounded-lg shadow p-4">
        <div className="flex flex-wrap gap-4">
          {/* Status Filter */}
          <div>
            <label className="block text-sm font-semibold text-gray-700 mb-2">Status</label>
            <div className="flex gap-2">
              {['all', 'unlocked', 'locked'].map(status => (
                <button
                  key={status}
                  onClick={() => setFilter(status)}
                  className={`px-4 py-2 rounded-lg font-medium text-sm transition-colors ${
                    filter === status
                      ? 'bg-blue-600 text-white'
                      : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                  }`}
                >
                  {status.charAt(0).toUpperCase() + status.slice(1)}
                </button>
              ))}
            </div>
          </div>

          {/* Category Filter */}
          <div>
            <label className="block text-sm font-semibold text-gray-700 mb-2">Category</label>
            <select
              value={categoryFilter}
              onChange={(e) => setCategoryFilter(e.target.value)}
              className="px-4 py-2 rounded-lg border border-gray-300 text-sm font-medium"
            >
              {categories.map(category => (
                <option key={category} value={category}>
                  {category.charAt(0).toUpperCase() + category.slice(1).replace('_', ' ')}
                </option>
              ))}
            </select>
          </div>

          {/* Tier Filter */}
          <div>
            <label className="block text-sm font-semibold text-gray-700 mb-2">Tier</label>
            <select
              value={tierFilter}
              onChange={(e) => setTierFilter(e.target.value)}
              className="px-4 py-2 rounded-lg border border-gray-300 text-sm font-medium"
            >
              {tiers.map(tier => (
                <option key={tier} value={tier}>
                  {tier.charAt(0).toUpperCase() + tier.slice(1)}
                </option>
              ))}
            </select>
          </div>

          {/* Results Count */}
          <div className="ml-auto flex items-end">
            <div className="text-sm text-gray-600">
              Showing {filteredAchievements.length} achievement{filteredAchievements.length !== 1 ? 's' : ''}
            </div>
          </div>
        </div>
      </div>

      {/* Achievement Grid */}
      <div className="bg-white rounded-lg shadow p-6">
        {filteredAchievements.length === 0 ? (
          <div className="text-center py-12">
            <div className="text-6xl mb-4">🔍</div>
            <p className="text-xl text-gray-600">No achievements found with current filters</p>
            <button
              onClick={() => {
                setFilter('all');
                setCategoryFilter('all');
                setTierFilter('all');
              }}
              className="mt-4 px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
            >
              Reset Filters
            </button>
          </div>
        ) : (
          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-6">
            {filteredAchievements.map(achievement => {
              const unlocked = isAchievementUnlocked(achievement.id);
              const unlockedAt = getUnlockDate(achievement.id);

              return (
                <div
                  key={achievement.id}
                  className="transform transition-transform hover:scale-105"
                >
                  <Badge
                    achievement={achievement}
                    unlocked={unlocked}
                    unlockedAt={unlockedAt}
                    size="medium"
                    showDetails={true}
                  />
                </div>
              );
            })}
          </div>
        )}
      </div>

      {/* Achievement Categories Breakdown */}
      <div className="bg-white rounded-lg shadow p-6">
        <h3 className="text-2xl font-bold text-gray-800 mb-4">Progress by Category</h3>
        <div className="space-y-4">
          {Object.entries(stats.byCategory).map(([category, total]) => {
            const unlocked = stats.unlockedByCategory[category] || 0;
            const percentage = Math.round((unlocked / total) * 100);

            return (
              <div key={category}>
                <div className="flex justify-between items-center mb-2">
                  <span className="font-semibold text-gray-700 capitalize">
                    {category.replace('_', ' ')}
                  </span>
                  <span className="text-sm text-gray-600">
                    {unlocked}/{total} ({percentage}%)
                  </span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div
                    className="bg-blue-600 h-2 rounded-full transition-all duration-500"
                    style={{ width: `${percentage}%` }}
                  />
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Close Button */}
      {onClose && (
        <div className="flex justify-center">
          <button
            onClick={onClose}
            className="px-8 py-3 bg-gray-600 text-white rounded-lg font-semibold hover:bg-gray-700"
          >
            Close
          </button>
        </div>
      )}
    </div>
  );
};

export default AchievementSystem;
