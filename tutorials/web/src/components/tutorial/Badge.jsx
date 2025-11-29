import React from 'react';

/**
 * Badge - Individual achievement badge component
 */
const Badge = ({ achievement, unlocked = false, unlockedAt = null, size = 'medium', showDetails = true }) => {
  // Size variants
  const sizes = {
    small: {
      container: 'w-16 h-16',
      icon: 'text-3xl',
      name: 'text-xs',
      points: 'text-xs'
    },
    medium: {
      container: 'w-24 h-24',
      icon: 'text-5xl',
      name: 'text-sm',
      points: 'text-sm'
    },
    large: {
      container: 'w-32 h-32',
      icon: 'text-7xl',
      name: 'text-base',
      points: 'text-base'
    }
  };

  // Tier colors
  const tierColors = {
    bronze: {
      bg: unlocked ? 'bg-gradient-to-br from-orange-400 to-orange-600' : 'bg-gray-300',
      border: unlocked ? 'border-orange-500' : 'border-gray-400',
      text: unlocked ? 'text-orange-900' : 'text-gray-600',
      glow: unlocked ? 'shadow-lg shadow-orange-500/50' : ''
    },
    silver: {
      bg: unlocked ? 'bg-gradient-to-br from-gray-300 to-gray-500' : 'bg-gray-300',
      border: unlocked ? 'border-gray-400' : 'border-gray-400',
      text: unlocked ? 'text-gray-900' : 'text-gray-600',
      glow: unlocked ? 'shadow-lg shadow-gray-500/50' : ''
    },
    gold: {
      bg: unlocked ? 'bg-gradient-to-br from-yellow-400 to-yellow-600' : 'bg-gray-300',
      border: unlocked ? 'border-yellow-500' : 'border-gray-400',
      text: unlocked ? 'text-yellow-900' : 'text-gray-600',
      glow: unlocked ? 'shadow-lg shadow-yellow-500/50' : ''
    },
    platinum: {
      bg: unlocked ? 'bg-gradient-to-br from-blue-300 via-purple-300 to-pink-300' : 'bg-gray-300',
      border: unlocked ? 'border-purple-500' : 'border-gray-400',
      text: unlocked ? 'text-purple-900' : 'text-gray-600',
      glow: unlocked ? 'shadow-xl shadow-purple-500/50' : ''
    }
  };

  const sizeStyle = sizes[size];
  const colorStyle = tierColors[achievement.tier];

  return (
    <div className="flex flex-col items-center">
      {/* Badge Circle */}
      <div
        className={`
          ${sizeStyle.container}
          ${colorStyle.bg}
          ${colorStyle.border}
          ${colorStyle.glow}
          rounded-full
          border-4
          flex items-center justify-center
          transition-all duration-300
          ${unlocked ? 'transform hover:scale-110' : 'opacity-50 grayscale'}
          relative
        `}
        title={achievement.description}
      >
        {/* Icon */}
        <span className={`${sizeStyle.icon} ${unlocked ? '' : 'opacity-40'}`}>
          {unlocked ? achievement.icon : '🔒'}
        </span>

        {/* Tier Badge */}
        {showDetails && unlocked && (
          <div className="absolute -top-2 -right-2 bg-white rounded-full px-2 py-1 text-xs font-bold border-2 border-gray-200 shadow">
            {achievement.tier.charAt(0).toUpperCase()}
          </div>
        )}
      </div>

      {/* Badge Info */}
      {showDetails && (
        <div className="mt-2 text-center max-w-[120px]">
          <div className={`font-bold ${sizeStyle.name} ${colorStyle.text}`}>
            {achievement.name}
          </div>
          {unlocked ? (
            <>
              <div className={`${sizeStyle.points} text-green-600 font-semibold mt-1`}>
                +{achievement.points} pts
              </div>
              {unlockedAt && (
                <div className="text-xs text-gray-500 mt-1">
                  {new Date(unlockedAt).toLocaleDateString()}
                </div>
              )}
            </>
          ) : (
            <div className={`${sizeStyle.name} text-gray-500 mt-1`}>
              {achievement.description}
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default Badge;
