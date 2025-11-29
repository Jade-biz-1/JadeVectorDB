import React, { useState, useEffect } from 'react';
import Badge from './Badge';

/**
 * AchievementNotification - Toast notification for achievement unlocks
 */
const AchievementNotification = ({ achievement, onClose, duration = 5000 }) => {
  const [visible, setVisible] = useState(false);
  const [closing, setClosing] = useState(false);

  useEffect(() => {
    // Animate in
    setTimeout(() => setVisible(true), 100);

    // Auto-close after duration
    const closeTimer = setTimeout(() => {
      handleClose();
    }, duration);

    return () => clearTimeout(closeTimer);
  }, [duration]);

  const handleClose = () => {
    setClosing(true);
    setTimeout(() => {
      setVisible(false);
      if (onClose) onClose();
    }, 300);
  };

  if (!achievement) return null;

  return (
    <div
      className={`
        fixed top-4 right-4 z-50
        transform transition-all duration-300 ease-out
        ${visible && !closing ? 'translate-x-0 opacity-100' : 'translate-x-full opacity-0'}
      `}
    >
      <div className="bg-white rounded-lg shadow-2xl border-2 border-yellow-400 overflow-hidden max-w-md">
        {/* Header with animation */}
        <div className="bg-gradient-to-r from-yellow-400 via-yellow-500 to-orange-500 px-4 py-2">
          <div className="flex items-center gap-2">
            <span className="text-2xl animate-bounce">🎉</span>
            <span className="text-white font-bold text-lg">Achievement Unlocked!</span>
            <button
              onClick={handleClose}
              className="ml-auto text-white hover:text-gray-200 text-xl"
            >
              ×
            </button>
          </div>
        </div>

        {/* Content */}
        <div className="p-4 flex items-center gap-4">
          {/* Badge */}
          <div className="flex-shrink-0 animate-pulse">
            <Badge
              achievement={achievement}
              unlocked={true}
              size="medium"
              showDetails={false}
            />
          </div>

          {/* Details */}
          <div className="flex-1">
            <h3 className="text-xl font-bold text-gray-800 mb-1">
              {achievement.name}
            </h3>
            <p className="text-sm text-gray-600 mb-2">
              {achievement.description}
            </p>
            <div className="flex items-center gap-4 text-sm">
              <span className="px-2 py-1 bg-yellow-100 text-yellow-800 rounded font-semibold">
                {achievement.tier.toUpperCase()}
              </span>
              <span className="text-green-600 font-bold">
                +{achievement.points} points
              </span>
            </div>
          </div>
        </div>

        {/* Progress bar */}
        <div className="h-1 bg-gray-200">
          <div
            className="h-1 bg-gradient-to-r from-yellow-400 to-orange-500 transition-all ease-linear"
            style={{
              width: closing ? '0%' : '100%',
              transitionDuration: `${duration}ms`
            }}
          />
        </div>
      </div>
    </div>
  );
};

/**
 * AchievementNotificationStack - Manages multiple notifications
 */
export const AchievementNotificationStack = ({ achievements = [] }) => {
  const [notifications, setNotifications] = useState([]);

  useEffect(() => {
    if (achievements.length > 0) {
      // Add new achievements to stack
      setNotifications(prev => [
        ...prev,
        ...achievements.map((achievement, index) => ({
          id: `${achievement.id}-${Date.now()}-${index}`,
          achievement,
          delay: index * 500 // Stagger notifications
        }))
      ]);
    }
  }, [achievements]);

  const removeNotification = (id) => {
    setNotifications(prev => prev.filter(n => n.id !== id));
  };

  return (
    <div className="fixed top-4 right-4 z-50 space-y-4">
      {notifications.map((notification, index) => (
        <div
          key={notification.id}
          style={{
            animationDelay: `${notification.delay}ms`
          }}
        >
          <AchievementNotification
            achievement={notification.achievement}
            onClose={() => removeNotification(notification.id)}
            duration={5000}
          />
        </div>
      ))}
    </div>
  );
};

export default AchievementNotification;
