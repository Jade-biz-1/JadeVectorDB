import React from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Alert, AlertDescription } from '@/components/ui/alert';
import {
  CheckCircle2,
  XCircle,
  Trophy,
  TrendingUp,
  Clock,
  RotateCcw,
  Download,
  ArrowRight
} from 'lucide-react';
import { Progress } from '@/components/ui/progress';

/**
 * QuizResults Component
 * Displays comprehensive quiz results with feedback and actions
 */
const QuizResults = ({
  scoreData,
  moduleTitle,
  timeSpent,
  feedback,
  onRetry,
  onViewAnswers,
  onContinue,
  statistics
}) => {
  const { earnedPoints, totalPoints, percentage, results } = scoreData;

  const getScoreColor = () => {
    if (percentage >= 90) return 'text-green-600 dark:text-green-400';
    if (percentage >= 70) return 'text-blue-600 dark:text-blue-400';
    if (percentage >= 50) return 'text-yellow-600 dark:text-yellow-400';
    return 'text-red-600 dark:text-red-400';
  };

  const getProgressColor = () => {
    if (percentage >= 90) return 'bg-green-600';
    if (percentage >= 70) return 'bg-blue-600';
    if (percentage >= 50) return 'bg-yellow-600';
    return 'bg-red-600';
  };

  const getFeedbackEmoji = () => {
    if (percentage >= 90) return '🎉';
    if (percentage >= 70) return '✅';
    if (percentage >= 50) return '📚';
    return '🔄';
  };

  const correctAnswers = results.filter(r => r.isCorrect).length;
  const totalQuestions = results.length;

  const formatTime = (seconds) => {
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = seconds % 60;
    return `${minutes}m ${remainingSeconds}s`;
  };

  const exportResults = () => {
    const exportData = {
      module: moduleTitle,
      date: new Date().toISOString(),
      score: `${percentage}%`,
      points: `${earnedPoints}/${totalPoints}`,
      correctAnswers: `${correctAnswers}/${totalQuestions}`,
      timeSpent: formatTime(timeSpent),
      passed: scoreData.passed,
      feedback: feedback.message,
      results: results.map(r => ({
        question: r.question,
        correct: r.isCorrect,
        points: `${r.points}/${r.maxPoints}`
      }))
    };

    const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `quiz-results-${moduleTitle.replace(/\s+/g, '-').toLowerCase()}-${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      {/* Main Score Card */}
      <Card className="border-2">
        <CardHeader className="text-center pb-4">
          <div className="flex justify-center mb-4">
            {scoreData.passed ? (
              <div className="w-24 h-24 rounded-full bg-green-100 dark:bg-green-950 flex items-center justify-center">
                <Trophy className="w-12 h-12 text-green-600 dark:text-green-400" />
              </div>
            ) : (
              <div className="w-24 h-24 rounded-full bg-gray-100 dark:bg-gray-800 flex items-center justify-center">
                <TrendingUp className="w-12 h-12 text-gray-600 dark:text-gray-400" />
              </div>
            )}
          </div>
          <CardTitle className="text-3xl mb-2">
            {scoreData.passed ? 'Congratulations!' : 'Quiz Complete'}
          </CardTitle>
          <CardDescription className="text-lg">
            {moduleTitle} Assessment
          </CardDescription>
        </CardHeader>

        <CardContent className="space-y-6">
          {/* Score Display */}
          <div className="text-center space-y-3">
            <div className={`text-6xl font-bold ${getScoreColor()}`}>
              {percentage}%
            </div>
            <div className="text-xl text-gray-600 dark:text-gray-400">
              {earnedPoints} out of {totalPoints} points
            </div>
            <Progress value={percentage} className="h-3" indicatorClassName={getProgressColor()} />
          </div>

          {/* Stats Grid */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 pt-4">
            <div className="text-center p-4 bg-gray-50 dark:bg-gray-900 rounded-lg">
              <div className="flex justify-center mb-2">
                <CheckCircle2 className="w-6 h-6 text-green-600 dark:text-green-400" />
              </div>
              <div className="text-2xl font-bold">{correctAnswers}</div>
              <div className="text-sm text-gray-600 dark:text-gray-400">Correct</div>
            </div>

            <div className="text-center p-4 bg-gray-50 dark:bg-gray-900 rounded-lg">
              <div className="flex justify-center mb-2">
                <XCircle className="w-6 h-6 text-red-600 dark:text-red-400" />
              </div>
              <div className="text-2xl font-bold">{totalQuestions - correctAnswers}</div>
              <div className="text-sm text-gray-600 dark:text-gray-400">Incorrect</div>
            </div>

            <div className="text-center p-4 bg-gray-50 dark:bg-gray-900 rounded-lg">
              <div className="flex justify-center mb-2">
                <Clock className="w-6 h-6 text-blue-600 dark:text-blue-400" />
              </div>
              <div className="text-2xl font-bold">{formatTime(timeSpent)}</div>
              <div className="text-sm text-gray-600 dark:text-gray-400">Time Spent</div>
            </div>
          </div>

          {/* Passing Status */}
          <Alert className={scoreData.passed ? 'border-green-500 bg-green-50 dark:bg-green-950' : 'border-yellow-500 bg-yellow-50 dark:bg-yellow-950'}>
            <AlertDescription className="flex items-center justify-between">
              <div>
                <span className="font-semibold">
                  {scoreData.passed ? '✅ Passed' : '📚 Not Passed Yet'}
                </span>
                <span className="text-sm ml-2">
                  (Passing score: 70%)
                </span>
              </div>
              {scoreData.passed && (
                <Badge variant="success" className="bg-green-600 text-white">
                  Certified
                </Badge>
              )}
            </AlertDescription>
          </Alert>
        </CardContent>
      </Card>

      {/* Feedback Card */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <span className="text-2xl">{getFeedbackEmoji()}</span>
            Performance Feedback
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <p className="text-lg font-medium">{feedback.message}</p>

          {feedback.suggestions && feedback.suggestions.length > 0 && (
            <div>
              <h4 className="font-semibold mb-3">💡 Suggestions for improvement:</h4>
              <ul className="space-y-2">
                {feedback.suggestions.map((suggestion, index) => (
                  <li key={index} className="flex items-start gap-2">
                    <ArrowRight className="w-4 h-4 mt-1 text-blue-600 dark:text-blue-400 flex-shrink-0" />
                    <span className="text-sm text-gray-700 dark:text-gray-300">
                      {suggestion}
                    </span>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Statistics Card (if available) */}
      {statistics && (
        <Card>
          <CardHeader>
            <CardTitle>Your Progress</CardTitle>
            <CardDescription>Historical performance for this module</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="text-center p-3 bg-gray-50 dark:bg-gray-900 rounded">
                <div className="text-xl font-bold">{statistics.attempts}</div>
                <div className="text-xs text-gray-600 dark:text-gray-400">Attempts</div>
              </div>
              <div className="text-center p-3 bg-gray-50 dark:bg-gray-900 rounded">
                <div className="text-xl font-bold">{statistics.bestScore}%</div>
                <div className="text-xs text-gray-600 dark:text-gray-400">Best Score</div>
              </div>
              <div className="text-center p-3 bg-gray-50 dark:bg-gray-900 rounded">
                <div className="text-xl font-bold">{statistics.averageScore}%</div>
                <div className="text-xs text-gray-600 dark:text-gray-400">Average</div>
              </div>
              <div className="text-center p-3 bg-gray-50 dark:bg-gray-900 rounded">
                <div className="text-xl font-bold">{formatTime(statistics.totalTimeSpent)}</div>
                <div className="text-xs text-gray-600 dark:text-gray-400">Total Time</div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Actions */}
      <div className="flex flex-col sm:flex-row gap-3 justify-center">
        <Button onClick={onViewAnswers} variant="outline" size="lg">
          📋 Review Answers
        </Button>

        <Button onClick={onRetry} variant="outline" size="lg">
          <RotateCcw className="w-4 h-4 mr-2" />
          Retake Quiz
        </Button>

        <Button onClick={exportResults} variant="outline" size="lg">
          <Download className="w-4 h-4 mr-2" />
          Export Results
        </Button>

        {scoreData.passed && onContinue && (
          <Button onClick={onContinue} size="lg" className="bg-green-600 hover:bg-green-700">
            Continue Learning →
          </Button>
        )}
      </div>

      {/* Detailed Results Summary */}
      <Card>
        <CardHeader>
          <CardTitle>Question Summary</CardTitle>
          <CardDescription>
            Overview of your answers for each question
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            {results.map((result, index) => (
              <div
                key={index}
                className={`flex items-start gap-3 p-3 rounded-lg border-2 ${
                  result.isCorrect
                    ? 'border-green-200 bg-green-50 dark:bg-green-950 dark:border-green-800'
                    : 'border-red-200 bg-red-50 dark:bg-red-950 dark:border-red-800'
                }`}
              >
                <div className="flex-shrink-0 mt-1">
                  {result.isCorrect ? (
                    <CheckCircle2 className="w-5 h-5 text-green-600 dark:text-green-400" />
                  ) : (
                    <XCircle className="w-5 h-5 text-red-600 dark:text-red-400" />
                  )}
                </div>
                <div className="flex-1 min-w-0">
                  <div className="font-medium text-sm mb-1">
                    Question {index + 1}
                  </div>
                  <div className="text-sm text-gray-700 dark:text-gray-300 line-clamp-2">
                    {result.question}
                  </div>
                </div>
                <div className="flex-shrink-0 text-right">
                  <div className={`font-bold ${result.isCorrect ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'}`}>
                    {result.points}/{result.maxPoints}
                  </div>
                  <div className="text-xs text-gray-500">points</div>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default QuizResults;
