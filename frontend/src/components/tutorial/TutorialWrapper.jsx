import React, { useState, useEffect } from 'react';
import { useContextualHelp } from '../../hooks/useContextualHelp';
import { checkAchievements } from '../../lib/achievementLogic';
import assessmentState from '../../lib/assessmentState';
import AssessmentSystem from './AssessmentSystem';
import ReadinessAssessment from './ReadinessAssessment';
import AchievementNotification from './AchievementNotification';
import HelpOverlay from './HelpOverlay';
import { HelpIcon } from './HelpTooltip';

/**
 * TutorialWrapper - Integration wrapper for tutorial modules
 *
 * This component demonstrates how to integrate:
 * - Module content
 * - Assessment system
 * - Achievement tracking
 * - Contextual help
 * - Readiness evaluation
 *
 * Usage:
 * <TutorialWrapper
 *   moduleId="module1"
 *   moduleName="Getting Started"
 *   onComplete={() => navigate('/next-module')}
 * >
 *   <YourModuleContent />
 * </TutorialWrapper>
 */
const TutorialWrapper = ({
  moduleId,
  moduleName,
  children,
  onComplete,
  showReadinessWhenComplete = false
}) => {
  const [showAssessment, setShowAssessment] = useState(false);
  const [showReadiness, setShowReadiness] = useState(false);
  const [moduleCompleted, setModuleCompleted] = useState(false);
  const [newAchievements, setNewAchievements] = useState([]);

  const { isHelpOpen, openHelp, closeHelp } = useContextualHelp();

  // Check if module is already passed
  useEffect(() => {
    const passed = assessmentState.hasPassedModule(moduleId);
    setModuleCompleted(passed);
  }, [moduleId]);

  /**
   * Handle module completion
   * Called when user finishes all module steps
   */
  const handleModuleComplete = () => {
    // Show assessment
    setShowAssessment(true);
  };

  /**
   * Handle assessment completion
   */
  const handleAssessmentComplete = (result) => {
    const { passed, score, timeSpent } = result;

    // Close assessment
    setShowAssessment(false);

    if (passed) {
      // Mark module as completed
      setModuleCompleted(true);

      // Check for achievements
      const context = {
        moduleId,
        score,
        timeSpent,
        passed: true
      };
      const unlocked = checkAchievements(context);

      if (unlocked.length > 0) {
        setNewAchievements(unlocked);
        // Clear after 6 seconds (5s display + 1s buffer)
        setTimeout(() => setNewAchievements([]), 6000);
      }

      // Check if all modules are complete
      const allModules = ['module1', 'module2', 'module3', 'module4', 'module5', 'module6'];
      const allComplete = allModules.every(id => assessmentState.hasPassedModule(id));

      if (allComplete && showReadinessWhenComplete) {
        // Show readiness assessment
        setTimeout(() => setShowReadiness(true), 1000);
      } else if (onComplete) {
        // Navigate to next module
        onComplete();
      }
    }
  };

  /**
   * Handle assessment retry
   */
  const handleAssessmentRetry = () => {
    setShowAssessment(true);
  };

  /**
   * Handle readiness assessment completion
   */
  const handleReadinessComplete = (evaluation) => {
    // Check for readiness-related achievements
    const context = {
      readinessLevel: evaluation.proficiencyLevel.label,
      overallScore: evaluation.overallScore
    };
    const unlocked = checkAchievements(context);

    if (unlocked.length > 0) {
      setNewAchievements(unlocked);
      setTimeout(() => setNewAchievements([]), 6000);
    }

    setShowReadiness(false);

    if (onComplete) {
      onComplete();
    }
  };

  // Render assessment if active
  if (showAssessment) {
    return (
      <>
        <AssessmentSystem
          moduleId={moduleId}
          onComplete={handleAssessmentComplete}
          onRetry={handleAssessmentRetry}
        />
        {newAchievements.length > 0 && (
          <AchievementNotification
            achievement={newAchievements[0]}
            onClose={() => setNewAchievements([])}
          />
        )}
        <HelpOverlay
          isOpen={isHelpOpen}
          onClose={closeHelp}
          initialContext="quiz-question"
        />
      </>
    );
  }

  // Render readiness assessment if active
  if (showReadiness) {
    return (
      <>
        <ReadinessAssessment
          onClose={() => setShowReadiness(false)}
          onContinue={handleReadinessComplete}
        />
        {newAchievements.length > 0 && (
          <AchievementNotification
            achievement={newAchievements[0]}
            onClose={() => setNewAchievements([])}
          />
        )}
        <HelpOverlay
          isOpen={isHelpOpen}
          onClose={closeHelp}
          initialContext="readiness-overview"
        />
      </>
    );
  }

  // Render module content
  return (
    <div className="tutorial-wrapper relative">
      {/* Help Button (Fixed Position) */}
      <div className="fixed bottom-6 right-6 z-40">
        <button
          onClick={() => openHelp()}
          className="w-14 h-14 bg-blue-600 hover:bg-blue-700 text-white rounded-full shadow-lg flex items-center justify-center transition-all hover:scale-110"
          aria-label="Open help"
        >
          <span className="text-2xl">?</span>
        </button>
      </div>

      {/* Module Status Banner */}
      {moduleCompleted && (
        <div className="mb-6 bg-green-50 border border-green-200 rounded-lg p-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <span className="text-3xl">✓</span>
              <div>
                <h3 className="font-bold text-green-900">Module Completed!</h3>
                <p className="text-green-700 text-sm">
                  You've passed the assessment for this module.
                </p>
              </div>
            </div>
            <button
              onClick={() => setShowAssessment(true)}
              className="px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 font-semibold text-sm"
            >
              Retake Assessment
            </button>
          </div>
        </div>
      )}

      {/* Module Header with Help */}
      <div className="mb-6 flex items-start justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-800 mb-2">{moduleName}</h1>
          <p className="text-gray-600">
            Complete all steps, then take the assessment to proceed
          </p>
        </div>
        <HelpIcon
          content="Use the help button (?) at the bottom right or press F1 anytime for assistance"
          title="Need Help?"
          position="left"
        />
      </div>

      {/* Module Content */}
      <div className="module-content">
        {children}
      </div>

      {/* Complete Module Button */}
      <div className="mt-8 flex justify-center">
        <button
          onClick={handleModuleComplete}
          className="px-8 py-4 bg-blue-600 text-white rounded-lg hover:bg-blue-700 font-bold text-lg shadow-lg hover:shadow-xl transition-all"
        >
          {moduleCompleted ? 'Retake Assessment' : 'Take Assessment'}
        </button>
      </div>

      {/* Achievement Notifications */}
      {newAchievements.length > 0 && (
        <div className="fixed top-4 right-4 z-50 space-y-4">
          {newAchievements.map((achievement, index) => (
            <AchievementNotification
              key={achievement.id}
              achievement={achievement}
              onClose={() => {
                setNewAchievements(prev => prev.filter((_, i) => i !== index));
              }}
            />
          ))}
        </div>
      )}

      {/* Help Overlay */}
      <HelpOverlay
        isOpen={isHelpOpen}
        onClose={closeHelp}
      />
    </div>
  );
};

/**
 * Example integration with GettingStarted module
 */
export const GettingStartedIntegrated = ({ onComplete }) => {
  // Import the original GettingStarted component content here
  // For demonstration purposes, we'll use a simplified version

  return (
    <TutorialWrapper
      moduleId="module1"
      moduleName="Module 1: Getting Started with JadeVectorDB"
      onComplete={onComplete}
    >
      {/* Your original GettingStarted module content goes here */}
      <div className="space-y-6">
        <div className="module-card">
          <h2 className="text-2xl font-bold text-gray-800 mb-4">
            Introduction to Vector Databases
          </h2>
          <div className="prose max-w-none">
            <p>
              A vector database is a specialized database designed to store, manage,
              and search high-dimensional vectors efficiently.
            </p>
            {/* ... rest of module content ... */}
          </div>
        </div>
      </div>
    </TutorialWrapper>
  );
};

export default TutorialWrapper;
