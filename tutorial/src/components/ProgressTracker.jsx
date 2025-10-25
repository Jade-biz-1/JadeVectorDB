import React from 'react';
import { useTutorialState } from '../hooks/useTutorialState';

const ProgressTracker = () => {
  const { state, overallProgress, moduleProgress, statistics } = useTutorialState();
  
  // Calculate completed modules
  const completedModules = state.modules.filter(module => 
    module.completedSteps === module.steps
  ).length;
  
  // Calculate in-progress modules
  const inProgressModules = state.modules.filter(module => 
    module.completedSteps > 0 && module.completedSteps < module.steps
  ).length;
  
  // Calculate locked modules
  const lockedModules = state.modules.filter(module => !module.unlocked).length;
  
  return (
    <div className="module-card">
      <div className="flex justify-between items-center mb-6">
        <h2 className="text-xl font-semibold text-gray-800">Tutorial Progress</h2>
        <div className="text-sm text-gray-600">
          {completedModules} of {state.modules.length} modules completed
        </div>
      </div>
      
      {/* Overall Progress */}
      <div className="mb-8">
        <div className="flex justify-between items-center mb-2">
          <h3 className="font-medium text-gray-800">Overall Progress</h3>
          <span className="text-lg font-bold text-blue-600">{overallProgress}%</span>
        </div>
        <div className="w-full bg-gray-200 rounded-full h-4">
          <div 
            className="bg-blue-600 h-4 rounded-full transition-all duration-500 ease-out"
            style={{ width: `${overallProgress}%` }}
          ></div>
        </div>
      </div>
      
      {/* Progress Statistics */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
        <div className="bg-blue-50 p-4 rounded-lg border border-blue-100">
          <div className="text-2xl font-bold text-blue-800">{statistics.totalModules}</div>
          <div className="text-sm text-blue-700">Total Modules</div>
        </div>
        <div className="bg-green-50 p-4 rounded-lg border border-green-100">
          <div className="text-2xl font-bold text-green-800">{statistics.achievementsUnlocked}</div>
          <div className="text-sm text-green-700">Achievements</div>
        </div>
        <div className="bg-purple-50 p-4 rounded-lg border border-purple-100">
          <div className="text-2xl font-bold text-purple-800">{Math.floor(statistics.totalTimeSpent / 3600)}</div>
          <div className="text-sm text-purple-700">Hours Spent</div>
        </div>
        <div className="bg-orange-50 p-4 rounded-lg border border-orange-100">
          <div className="text-2xl font-bold text-orange-800">{statistics.currentStreak}</div>
          <div className="text-sm text-orange-700">Day Streak</div>
        </div>
      </div>
      
      {/* Module Progress */}
      <div className="mb-6">
        <h3 className="font-medium text-gray-800 mb-4">Module Progress</h3>
        <div className="space-y-4">
          {state.modules.map((module, index) => {
            const progressPercent = moduleProgress(index);
            const isCurrent = index === state.currentModule;
            
            return (
              <div 
                key={module.id} 
                className={`p-4 rounded-lg border transition-all ${
                  isCurrent 
                    ? 'border-blue-300 bg-blue-50' 
                    : module.completedSteps === module.steps
                    ? 'border-green-200 bg-green-50'
                    : 'border-gray-200 hover:bg-gray-50'
                }`}
              >
                <div className="flex justify-between items-start mb-2">
                  <div>
                    <h4 className={`font-medium ${
                      isCurrent ? 'text-blue-800' : 
                      module.completedSteps === module.steps ? 'text-green-800' : 'text-gray-800'
                    }`}>
                      {module.title}
                    </h4>
                    <p className="text-sm text-gray-600 mt-1">{module.description}</p>
                  </div>
                  <div className="text-right">
                    <div className={`text-sm font-medium ${
                      module.completedSteps === module.steps ? 'text-green-600' : 'text-gray-600'
                    }`}>
                      {module.completedSteps}/{module.steps} steps
                    </div>
                    {module.completedSteps === module.steps && (
                      <div className="text-xs text-green-600 mt-1">Completed</div>
                    )}
                  </div>
                </div>
                
                <div className="w-full bg-gray-200 rounded-full h-2 mt-2">
                  <div 
                    className={`h-2 rounded-full transition-all duration-500 ease-out ${
                      module.completedSteps === module.steps 
                        ? 'bg-green-500' 
                        : isCurrent 
                        ? 'bg-blue-500' 
                        : 'bg-gray-400'
                    }`}
                    style={{ width: `${progressPercent}%` }}
                  ></div>
                </div>
                
                {module.completionDate && (
                  <div className="text-xs text-green-600 mt-2">
                    Completed on {new Date(module.completionDate).toLocaleDateString()}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>
      
      {/* Achievements Preview */}
      {state.achievements.length > 0 && (
        <div className="mb-6">
          <h3 className="font-medium text-gray-800 mb-3">Recent Achievements</h3>
          <div className="flex flex-wrap gap-2">
            {state.achievements.slice(0, 5).map((achievement, index) => (
              <div 
                key={index} 
                className="inline-flex items-center px-3 py-1 rounded-full text-xs font-medium bg-yellow-100 text-yellow-800 border border-yellow-200"
              >
                <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4M7.835 4.697a3.42 3.42 0 001.946-.806 3.42 3.42 0 014.438 0 3.42 3.42 0 001.946.806 3.42 3.42 0 013.138 3.138 3.42 3.42 0 00.806 1.946 3.42 3.42 0 010 4.438 3.42 3.42 0 00-.806 1.946 3.42 3.42 0 01-3.138 3.138 3.42 3.42 0 00-1.946.806 3.42 3.42 0 01-4.438 0 3.42 3.42 0 00-1.946-.806 3.42 3.42 0 01-3.138-3.138 3.42 3.42 0 00-.806-1.946 3.42 3.42 0 010-4.438 3.42 3.42 0 00.806-1.946 3.42 3.42 0 013.138-3.138z" />
                </svg>
                {achievement.title}
              </div>
            ))}
            {state.achievements.length > 5 && (
              <div className="inline-flex items-center px-3 py-1 rounded-full text-xs font-medium bg-gray-100 text-gray-800 border border-gray-200">
                +{state.achievements.length - 5} more
              </div>
            )}
          </div>
        </div>
      )}
      
      {/* Action Buttons */}
      <div className="flex flex-wrap gap-3 pt-4 border-t border-gray-200">
        <button className="btn-secondary text-sm">
          Export Progress
        </button>
        <button className="btn-secondary text-sm">
          Reset Progress
        </button>
        <button className="btn-primary text-sm">
          Continue Learning
        </button>
      </div>
    </div>
  );
};

export default ProgressTracker;