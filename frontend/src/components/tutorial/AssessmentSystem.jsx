import React, { useState, useEffect, useCallback } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Progress } from '@/components/ui/progress';
import {
  BookOpen,
  Clock,
  Award,
  AlertTriangle,
  PlayCircle,
  PauseCircle,
  CheckCircle
} from 'lucide-react';
import QuizQuestion from './QuizQuestion';
import QuizResults from './QuizResults';
import assessmentEngine from '@/lib/assessmentEngine';
import quizData from '@/data/quizQuestions.json';

/**
 * AssessmentSystem Component
 * Main orchestrator for the quiz/assessment system
 */
const AssessmentSystem = ({ moduleId, onComplete }) => {
  const [quizState, setQuizState] = useState('selection'); // selection, in-progress, paused, completed, review
  const [currentQuestionIndex, setCurrentQuestionIndex] = useState(0);
  const [userAnswers, setUserAnswers] = useState([]);
  const [startTime, setStartTime] = useState(null);
  const [timeRemaining, setTimeRemaining] = useState(null);
  const [scoreData, setScoreData] = useState(null);
  const [showResults, setShowResults] = useState(false);
  const [reviewMode, setReviewMode] = useState(false);

  const moduleQuiz = quizData.modules[moduleId];
  const questions = moduleQuiz?.questions || [];
  const timeLimit = moduleQuiz?.timeLimit || 600; // Default 10 minutes

  // Load saved progress on mount
  useEffect(() => {
    const savedProgress = assessmentEngine.loadProgress();
    if (savedProgress && savedProgress.moduleId === moduleId) {
      // Ask user if they want to resume
      setQuizState('paused');
      setCurrentQuestionIndex(savedProgress.currentQuestionIndex);
      setUserAnswers(savedProgress.userAnswers);
      setStartTime(savedProgress.startTime);
    }
  }, [moduleId]);

  // Timer effect
  useEffect(() => {
    if (quizState === 'in-progress' && startTime) {
      const timer = setInterval(() => {
        const remaining = assessmentEngine.getRemainingTime(startTime, timeLimit);
        setTimeRemaining(remaining);

        if (remaining <= 0) {
          handleTimeExpired();
        }
      }, 1000);

      return () => clearInterval(timer);
    }
  }, [quizState, startTime, timeLimit]);

  // Auto-save progress
  useEffect(() => {
    if (quizState === 'in-progress' && startTime) {
      const interval = setInterval(() => {
        assessmentEngine.saveProgress(moduleId, currentQuestionIndex, userAnswers, startTime);
      }, 30000); // Save every 30 seconds

      return () => clearInterval(interval);
    }
  }, [quizState, moduleId, currentQuestionIndex, userAnswers, startTime]);

  const handleStartQuiz = () => {
    const now = Date.now();
    setQuizState('in-progress');
    setStartTime(now);
    setTimeRemaining(timeLimit);
    setCurrentQuestionIndex(0);
    setUserAnswers(new Array(questions.length).fill(null));
    assessmentEngine.clearProgress();
  };

  const handleResumeQuiz = () => {
    setQuizState('in-progress');
  };

  const handlePauseQuiz = () => {
    setQuizState('paused');
    assessmentEngine.saveProgress(moduleId, currentQuestionIndex, userAnswers, startTime);
  };

  const handleAnswerChange = (answer) => {
    const newAnswers = [...userAnswers];
    newAnswers[currentQuestionIndex] = answer;
    setUserAnswers(newAnswers);
  };

  const handleNext = () => {
    if (currentQuestionIndex < questions.length - 1) {
      setCurrentQuestionIndex(currentQuestionIndex + 1);
    }
  };

  const handlePrevious = () => {
    if (currentQuestionIndex > 0) {
      setCurrentQuestionIndex(currentQuestionIndex - 1);
    }
  };

  const handleSubmitQuiz = useCallback(() => {
    const timeSpent = Math.floor((Date.now() - startTime) / 1000);
    const score = assessmentEngine.calculateScore(questions, userAnswers);
    const feedback = assessmentEngine.generateFeedback(score.percentage, moduleId);
    const statistics = assessmentEngine.getStatistics(moduleId);

    // Save results
    assessmentEngine.saveResults(moduleId, score, timeSpent);
    assessmentEngine.clearProgress();

    setScoreData({
      ...score,
      timeSpent,
      feedback,
      statistics
    });
    setQuizState('completed');
    setShowResults(true);

    // Notify parent component
    if (onComplete) {
      onComplete({
        passed: score.passed,
        percentage: score.percentage,
        timeSpent
      });
    }
  }, [startTime, questions, userAnswers, moduleId, onComplete]);

  const handleTimeExpired = useCallback(() => {
    handleSubmitQuiz();
  }, [handleSubmitQuiz]);

  const handleRetry = () => {
    setQuizState('selection');
    setCurrentQuestionIndex(0);
    setUserAnswers([]);
    setStartTime(null);
    setTimeRemaining(null);
    setScoreData(null);
    setShowResults(false);
    setReviewMode(false);
  };

  const handleViewAnswers = () => {
    setReviewMode(true);
    setQuizState('review');
    setCurrentQuestionIndex(0);
  };

  const formatTime = (seconds) => {
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = seconds % 60;
    return `${minutes.toString().padStart(2, '0')}:${remainingSeconds.toString().padStart(2, '0')}`;
  };

  const getTimeColor = () => {
    if (timeRemaining > 120) return 'text-green-600 dark:text-green-400';
    if (timeRemaining > 60) return 'text-yellow-600 dark:text-yellow-400';
    return 'text-red-600 dark:text-red-400';
  };

  const getAnsweredCount = () => {
    return userAnswers.filter(a => a !== null && a !== undefined && a !== '').length;
  };

  const renderQuizSelection = () => {
    const stats = assessmentEngine.getStatistics(moduleId);
    const hasAttempts = stats.moduleStats[moduleId]?.attempts > 0;

    return (
      <div className="max-w-3xl mx-auto space-y-6">
        <Card>
          <CardHeader className="text-center">
            <div className="flex justify-center mb-4">
              <div className="w-20 h-20 rounded-full bg-blue-100 dark:bg-blue-950 flex items-center justify-center">
                <BookOpen className="w-10 h-10 text-blue-600 dark:text-blue-400" />
              </div>
            </div>
            <CardTitle className="text-2xl">{moduleQuiz.title}</CardTitle>
            <CardDescription className="text-base">
              Test your knowledge of this module with interactive questions
            </CardDescription>
          </CardHeader>

          <CardContent className="space-y-6">
            {/* Quiz Info */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="text-center p-4 bg-gray-50 dark:bg-gray-900 rounded-lg">
                <div className="text-2xl font-bold text-blue-600 dark:text-blue-400">
                  {questions.length}
                </div>
                <div className="text-sm text-gray-600 dark:text-gray-400">Questions</div>
              </div>

              <div className="text-center p-4 bg-gray-50 dark:bg-gray-900 rounded-lg">
                <div className="text-2xl font-bold text-blue-600 dark:text-blue-400">
                  {formatTime(timeLimit)}
                </div>
                <div className="text-sm text-gray-600 dark:text-gray-400">Time Limit</div>
              </div>

              <div className="text-center p-4 bg-gray-50 dark:bg-gray-900 rounded-lg">
                <div className="text-2xl font-bold text-blue-600 dark:text-blue-400">
                  {moduleQuiz.passingScore}%
                </div>
                <div className="text-sm text-gray-600 dark:text-gray-400">Passing Score</div>
              </div>
            </div>

            {/* Previous Attempts */}
            {hasAttempts && (
              <Alert className="border-blue-500">
                <AlertDescription>
                  <div className="font-semibold mb-2">📊 Your Previous Performance</div>
                  <div className="grid grid-cols-2 gap-2 text-sm">
                    <div>Attempts: {stats.moduleStats[moduleId].attempts}</div>
                    <div>Best Score: {stats.moduleStats[moduleId].bestScore}%</div>
                    <div>Average: {stats.moduleStats[moduleId].averageScore}%</div>
                    <div>Status: {stats.moduleStats[moduleId].passed ? '✅ Passed' : '📚 In Progress'}</div>
                  </div>
                </AlertDescription>
              </Alert>
            )}

            {/* Instructions */}
            <div className="space-y-3">
              <h4 className="font-semibold">📋 Instructions:</h4>
              <ul className="space-y-2 text-sm text-gray-700 dark:text-gray-300">
                <li>• Answer all questions to the best of your ability</li>
                <li>• You can navigate between questions using Previous/Next buttons</li>
                <li>• Your progress is auto-saved every 30 seconds</li>
                <li>• You must score {moduleQuiz.passingScore}% or higher to pass</li>
                <li>• You can retake the quiz as many times as needed</li>
              </ul>
            </div>

            <Button onClick={handleStartQuiz} size="lg" className="w-full">
              <PlayCircle className="w-5 h-5 mr-2" />
              {hasAttempts ? 'Retake Quiz' : 'Start Quiz'}
            </Button>
          </CardContent>
        </Card>
      </div>
    );
  };

  const renderPausedState = () => {
    const answeredCount = getAnsweredCount();
    const progress = (answeredCount / questions.length) * 100;

    return (
      <div className="max-w-3xl mx-auto space-y-6">
        <Card>
          <CardHeader className="text-center">
            <div className="flex justify-center mb-4">
              <div className="w-20 h-20 rounded-full bg-yellow-100 dark:bg-yellow-950 flex items-center justify-center">
                <PauseCircle className="w-10 h-10 text-yellow-600 dark:text-yellow-400" />
              </div>
            </div>
            <CardTitle className="text-2xl">Quiz Paused</CardTitle>
            <CardDescription>
              Your progress has been saved. You can resume anytime.
            </CardDescription>
          </CardHeader>

          <CardContent className="space-y-6">
            <div className="space-y-3">
              <div className="flex justify-between text-sm">
                <span>Progress</span>
                <span className="font-medium">
                  {answeredCount} of {questions.length} answered
                </span>
              </div>
              <Progress value={progress} className="h-2" />
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div className="text-center p-4 bg-gray-50 dark:bg-gray-900 rounded-lg">
                <div className="text-xl font-bold">Question {currentQuestionIndex + 1}</div>
                <div className="text-xs text-gray-600 dark:text-gray-400">Current Position</div>
              </div>

              <div className="text-center p-4 bg-gray-50 dark:bg-gray-900 rounded-lg">
                <div className="text-xl font-bold">
                  {timeRemaining ? formatTime(timeRemaining) : '--:--'}
                </div>
                <div className="text-xs text-gray-600 dark:text-gray-400">Time Remaining</div>
              </div>
            </div>

            <div className="flex gap-3">
              <Button onClick={handleResumeQuiz} size="lg" className="flex-1">
                <PlayCircle className="w-5 h-5 mr-2" />
                Resume Quiz
              </Button>
              <Button onClick={handleRetry} variant="outline" size="lg" className="flex-1">
                Start Over
              </Button>
            </div>
          </CardContent>
        </Card>
      </div>
    );
  };

  const renderQuizInProgress = () => {
    const answeredCount = getAnsweredCount();
    const progress = ((currentQuestionIndex + 1) / questions.length) * 100;
    const currentQuestion = questions[currentQuestionIndex];

    return (
      <div className="max-w-4xl mx-auto space-y-4">
        {/* Progress Header */}
        <Card>
          <CardContent className="py-4">
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center gap-4">
                <Badge variant="outline" className="text-sm">
                  {answeredCount} / {questions.length} Answered
                </Badge>
                {timeRemaining !== null && (
                  <div className={`flex items-center gap-2 font-mono text-lg font-bold ${getTimeColor()}`}>
                    <Clock className="w-5 h-5" />
                    {formatTime(timeRemaining)}
                  </div>
                )}
              </div>
              <Button onClick={handlePauseQuiz} variant="outline" size="sm">
                <PauseCircle className="w-4 h-4 mr-2" />
                Pause
              </Button>
            </div>
            <Progress value={progress} className="h-2" />
          </CardContent>
        </Card>

        {/* Warning for low time */}
        {timeRemaining !== null && timeRemaining <= 60 && (
          <Alert variant="destructive">
            <AlertTriangle className="w-4 h-4" />
            <AlertDescription>
              <span className="font-semibold">Time Warning:</span> Only {timeRemaining} seconds remaining!
            </AlertDescription>
          </Alert>
        )}

        {/* Question */}
        <QuizQuestion
          question={currentQuestion}
          questionNumber={currentQuestionIndex + 1}
          totalQuestions={questions.length}
          userAnswer={userAnswers[currentQuestionIndex]}
          onAnswerChange={handleAnswerChange}
          onNext={handleNext}
          onPrevious={handlePrevious}
          onSubmit={handleSubmitQuiz}
          isFirstQuestion={currentQuestionIndex === 0}
          isLastQuestion={currentQuestionIndex === questions.length - 1}
        />
      </div>
    );
  };

  const renderReviewMode = () => {
    if (!scoreData) return null;

    const currentQuestion = questions[currentQuestionIndex];
    const currentResult = scoreData.results[currentQuestionIndex];

    return (
      <div className="max-w-4xl mx-auto space-y-4">
        <Card>
          <CardContent className="py-4">
            <div className="flex items-center justify-between">
              <div>
                <Badge variant={currentResult.isCorrect ? 'success' : 'destructive'}>
                  {currentResult.isCorrect ? '✅ Correct' : '❌ Incorrect'}
                </Badge>
                <span className="ml-3 text-sm text-gray-600 dark:text-gray-400">
                  {currentResult.points} / {currentResult.maxPoints} points
                </span>
              </div>
              <Button onClick={() => setQuizState('completed')} variant="outline" size="sm">
                Back to Results
              </Button>
            </div>
          </CardContent>
        </Card>

        <QuizQuestion
          question={currentQuestion}
          questionNumber={currentQuestionIndex + 1}
          totalQuestions={questions.length}
          userAnswer={userAnswers[currentQuestionIndex]}
          onAnswerChange={() => {}}
          onNext={handleNext}
          onPrevious={handlePrevious}
          showResults={true}
          isCorrect={currentResult.isCorrect}
          isFirstQuestion={currentQuestionIndex === 0}
          isLastQuestion={currentQuestionIndex === questions.length - 1}
        />
      </div>
    );
  };

  // Render based on state
  switch (quizState) {
    case 'selection':
      return renderQuizSelection();
    case 'paused':
      return renderPausedState();
    case 'in-progress':
      return renderQuizInProgress();
    case 'completed':
      return (
        <QuizResults
          scoreData={scoreData}
          moduleTitle={moduleQuiz.title}
          timeSpent={scoreData.timeSpent}
          feedback={scoreData.feedback}
          onRetry={handleRetry}
          onViewAnswers={handleViewAnswers}
          onContinue={onComplete}
          statistics={scoreData.statistics?.moduleStats?.[moduleId]}
        />
      );
    case 'review':
      return renderReviewMode();
    default:
      return null;
  }
};

export default AssessmentSystem;
