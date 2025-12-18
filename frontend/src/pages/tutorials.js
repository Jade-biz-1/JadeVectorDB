import React, { useState } from 'react';
import Head from 'next/head';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Alert, AlertDescription } from '@/components/ui/alert';
import {
  BookOpen,
  CheckCircle,
  Trophy,
  Star,
  HelpCircle,
  ArrowRight,
  Play
} from 'lucide-react';
import TutorialWrapper from '@/components/tutorial/TutorialWrapper';
import AssessmentSystem from '@/components/tutorial/AssessmentSystem';
import ReadinessAssessment from '@/components/tutorial/ReadinessAssessment';
import AchievementSystem from '@/components/tutorial/AchievementSystem';
import assessmentState from '@/lib/assessmentState';
import { getUnlockedAchievements } from '@/lib/achievementLogic';

/**
 * Tutorials Page - Sprint 5 Tutorial Enhancement Demo
 * 
 * Features:
 * - Achievement/Badge System (T215.14)
 * - Contextual Help System (T215.15)
 * - Hint System (T215.16)
 * - Assessment System (T215.21)
 * - Readiness Assessment (T215.24)
 */
export default function Tutorials() {
  const [selectedModule, setSelectedModule] = useState(null);
  const [showAssessment, setShowAssessment] = useState(false);
  const [showReadiness, setShowReadiness] = useState(false);
  const [showAchievements, setShowAchievements] = useState(false);

  // Tutorial modules
  const modules = [
    {
      id: 'module1',
      title: 'Getting Started',
      description: 'Introduction to vector databases and JadeVectorDB basics',
      duration: '15 min',
      difficulty: 'Beginner',
      icon: '🚀'
    },
    {
      id: 'module2',
      title: 'Vector Operations',
      description: 'Learn how to store, retrieve, and manipulate vectors',
      duration: '20 min',
      difficulty: 'Beginner',
      icon: '🔢'
    },
    {
      id: 'module3',
      title: 'Advanced Search',
      description: 'Master similarity search and ranking algorithms',
      duration: '25 min',
      difficulty: 'Intermediate',
      icon: '🔍'
    },
    {
      id: 'module4',
      title: 'Metadata Filtering',
      description: 'Combine vector search with metadata filtering',
      duration: '20 min',
      difficulty: 'Intermediate',
      icon: '🎛️'
    },
    {
      id: 'module5',
      title: 'Batch Operations',
      description: 'Efficiently handle large-scale vector operations',
      duration: '25 min',
      difficulty: 'Advanced',
      icon: '⚡'
    },
    {
      id: 'module6',
      title: 'Production Best Practices',
      description: 'Learn production deployment and optimization strategies',
      duration: '30 min',
      difficulty: 'Advanced',
      icon: '🏭'
    }
  ];

  // Get progress statistics
  const getModuleProgress = (moduleId) => {
    const passed = assessmentState.hasPassedModule(moduleId);
    const bestScore = assessmentState.getBestScore(moduleId);
    const attempts = assessmentState.getModuleHistory(moduleId).length;
    return { passed, bestScore, attempts };
  };

  // Check if all modules completed
  const allModulesCompleted = modules.every(m => 
    assessmentState.hasPassedModule(m.id)
  );

  // Get unlocked achievements
  const unlockedAchievements = getUnlockedAchievements();

  const handleModuleSelect = (moduleId) => {
    setSelectedModule(moduleId);
  };

  const handleStartAssessment = (moduleId) => {
    setSelectedModule(moduleId);
    setShowAssessment(true);
  };

  const handleAssessmentComplete = (result) => {
    setShowAssessment(false);
    setSelectedModule(null);
    // Refresh page to show updated progress
  };

  const handleStartReadiness = () => {
    setShowReadiness(true);
  };

  if (showReadiness) {
    return (
      <>
        <Head>
          <title>Readiness Assessment - JadeVectorDB Tutorials</title>
        </Head>
        <div className="min-h-screen bg-gray-50 dark:bg-gray-900 p-6">
          <div className="max-w-6xl mx-auto">
            <Button
              variant="outline"
              onClick={() => setShowReadiness(false)}
              className="mb-6"
            >
              ← Back to Tutorials
            </Button>
            <ReadinessAssessment
              onComplete={(result) => {
                setShowReadiness(false);
                // Could show certificate or results
              }}
            />
          </div>
        </div>
      </>
    );
  }

  if (showAchievements) {
    return (
      <>
        <Head>
          <title>Achievements - JadeVectorDB Tutorials</title>
        </Head>
        <div className="min-h-screen bg-gray-50 dark:bg-gray-900 p-6">
          <div className="max-w-6xl mx-auto">
            <Button
              variant="outline"
              onClick={() => setShowAchievements(false)}
              className="mb-6"
            >
              ← Back to Tutorials
            </Button>
            <AchievementSystem />
          </div>
        </div>
      </>
    );
  }

  if (showAssessment && selectedModule) {
    return (
      <>
        <Head>
          <title>Module Assessment - JadeVectorDB Tutorials</title>
        </Head>
        <div className="min-h-screen bg-gray-50 dark:bg-gray-900 p-6">
          <div className="max-w-6xl mx-auto">
            <Button
              variant="outline"
              onClick={() => {
                setShowAssessment(false);
                setSelectedModule(null);
              }}
              className="mb-6"
            >
              ← Back to Tutorials
            </Button>
            <AssessmentSystem
              moduleId={selectedModule}
              onComplete={handleAssessmentComplete}
              minPassScore={70}
            />
          </div>
        </div>
      </>
    );
  }

  return (
    <>
      <Head>
        <title>Interactive Tutorials - JadeVectorDB</title>
        <meta name="description" content="Learn JadeVectorDB with interactive tutorials, assessments, and achievements" />
      </Head>

      <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
        {/* Header */}
        <div className="bg-white dark:bg-gray-800 border-b">
          <div className="max-w-7xl mx-auto px-6 py-8">
            <div className="flex items-center justify-between">
              <div>
                <h1 className="text-4xl font-bold mb-2">
                  JadeVectorDB Tutorials
                </h1>
                <p className="text-gray-600 dark:text-gray-400">
                  Master vector databases with interactive lessons, quizzes, and achievements
                </p>
              </div>
              <div className="flex gap-3">
                <Button
                  variant="outline"
                  onClick={() => setShowAchievements(true)}
                  className="flex items-center gap-2"
                >
                  <Trophy className="w-4 h-4" />
                  Achievements ({unlockedAchievements.length})
                </Button>
                {allModulesCompleted && (
                  <Button
                    onClick={handleStartReadiness}
                    className="flex items-center gap-2"
                  >
                    <Star className="w-4 h-4" />
                    Final Assessment
                  </Button>
                )}
              </div>
            </div>

            {/* Progress Summary */}
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mt-6">
              <Card>
                <CardContent className="pt-6">
                  <div className="text-center">
                    <div className="text-3xl font-bold text-blue-600 dark:text-blue-400">
                      {modules.filter(m => getModuleProgress(m.id).passed).length}/{modules.length}
                    </div>
                    <div className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                      Modules Completed
                    </div>
                  </div>
                </CardContent>
              </Card>
              <Card>
                <CardContent className="pt-6">
                  <div className="text-center">
                    <div className="text-3xl font-bold text-green-600 dark:text-green-400">
                      {unlockedAchievements.length}
                    </div>
                    <div className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                      Achievements Unlocked
                    </div>
                  </div>
                </CardContent>
              </Card>
              <Card>
                <CardContent className="pt-6">
                  <div className="text-center">
                    <div className="text-3xl font-bold text-purple-600 dark:text-purple-400">
                      {unlockedAchievements.reduce((sum, a) => sum + (a.points || 0), 0)}
                    </div>
                    <div className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                      Total Points
                    </div>
                  </div>
                </CardContent>
              </Card>
              <Card>
                <CardContent className="pt-6">
                  <div className="text-center">
                    <div className="text-3xl font-bold text-orange-600 dark:text-orange-400">
                      {allModulesCompleted ? '100%' : 
                        `${Math.round((modules.filter(m => getModuleProgress(m.id).passed).length / modules.length) * 100)}%`
                      }
                    </div>
                    <div className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                      Overall Progress
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>
          </div>
        </div>

        {/* Module Grid */}
        <div className="max-w-7xl mx-auto px-6 py-8">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {modules.map((module) => {
              const progress = getModuleProgress(module.id);
              
              return (
                <Card key={module.id} className="hover:shadow-lg transition-shadow">
                  <CardHeader>
                    <div className="flex items-start justify-between">
                      <div className="text-4xl mb-2">{module.icon}</div>
                      {progress.passed && (
                        <Badge variant="success" className="bg-green-100 text-green-800">
                          <CheckCircle className="w-3 h-3 mr-1" />
                          Completed
                        </Badge>
                      )}
                    </div>
                    <CardTitle className="text-xl">{module.title}</CardTitle>
                    <CardDescription>{module.description}</CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    {/* Module Info */}
                    <div className="flex items-center gap-4 text-sm text-gray-600 dark:text-gray-400">
                      <span>⏱️ {module.duration}</span>
                      <span>📊 {module.difficulty}</span>
                    </div>

                    {/* Progress Info */}
                    {progress.attempts > 0 && (
                      <Alert>
                        <AlertDescription>
                          <div className="text-sm">
                            <div>Best Score: <strong>{progress.bestScore}%</strong></div>
                            <div>Attempts: {progress.attempts}</div>
                          </div>
                        </AlertDescription>
                      </Alert>
                    )}

                    {/* Actions */}
                    <div className="flex gap-2">
                      <Button
                        onClick={() => handleStartAssessment(module.id)}
                        className="flex-1 flex items-center justify-center gap-2"
                        variant={progress.passed ? "outline" : "default"}
                      >
                        {progress.passed ? (
                          <>
                            <Play className="w-4 h-4" />
                            Retake Quiz
                          </>
                        ) : (
                          <>
                            <Play className="w-4 h-4" />
                            Start Quiz
                          </>
                        )}
                      </Button>
                    </div>
                  </CardContent>
                </Card>
              );
            })}
          </div>

          {/* Sprint 5 Features Notice */}
          <Card className="mt-8 bg-blue-50 dark:bg-blue-950 border-blue-200">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Star className="w-5 h-5 text-blue-600" />
                New: Sprint 5 Tutorial Enhancements
              </CardTitle>
            </CardHeader>
            <CardContent>
              <ul className="space-y-2 text-sm">
                <li className="flex items-center gap-2">
                  <Trophy className="w-4 h-4 text-yellow-600" />
                  <strong>Achievement System:</strong> Earn badges and points for completing modules
                </li>
                <li className="flex items-center gap-2">
                  <HelpCircle className="w-4 h-4 text-green-600" />
                  <strong>Contextual Help:</strong> Press F1 or ? for instant help anywhere
                </li>
                <li className="flex items-center gap-2">
                  <BookOpen className="w-4 h-4 text-purple-600" />
                  <strong>Progressive Hints:</strong> Get hints when stuck on quiz questions
                </li>
                <li className="flex items-center gap-2">
                  <CheckCircle className="w-4 h-4 text-blue-600" />
                  <strong>Assessments:</strong> Test your knowledge with interactive quizzes
                </li>
                <li className="flex items-center gap-2">
                  <Star className="w-4 h-4 text-orange-600" />
                  <strong>Readiness Assessment:</strong> Final exam to validate your skills
                </li>
              </ul>
            </CardContent>
          </Card>
        </div>
      </div>
    </>
  );
}
