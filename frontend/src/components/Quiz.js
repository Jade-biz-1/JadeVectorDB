import { useState, useEffect } from 'react';
import { Button } from './ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Alert, AlertTitle, AlertDescription } from './ui/alert';
import assessmentEngine from '../lib/assessmentEngine';

export default function Quiz({ quizData, onComplete }) {
  const [currentQuestionIndex, setCurrentQuestionIndex] = useState(0);
  const [userAnswers, setUserAnswers] = useState([]);
  const [startTime, setStartTime] = useState(null);
  const [timeRemaining, setTimeRemaining] = useState(null);
  const [showResults, setShowResults] = useState(false);
  const [scoreData, setScoreData] = useState(null);
  const [loading, setLoading] = useState(false);

  // Initialize quiz
  useEffect(() => {
    if (!quizData) return;

    // Check for saved progress
    const savedProgress = assessmentEngine.loadProgress();
    if (savedProgress && savedProgress.moduleId === quizData.id) {
      const useProgress = confirm(
        'You have saved progress for this quiz. Would you like to continue where you left off?'
      );

      if (useProgress) {
        setCurrentQuestionIndex(savedProgress.currentQuestionIndex);
        setUserAnswers(savedProgress.userAnswers);
        setStartTime(savedProgress.startTime);
      } else {
        startNewQuiz();
      }
    } else {
      startNewQuiz();
    }
  }, [quizData]);

  // Timer countdown
  useEffect(() => {
    if (!startTime || !quizData?.timeLimit || showResults) return;

    const interval = setInterval(() => {
      const remaining = assessmentEngine.getRemainingTime(startTime, quizData.timeLimit);
      setTimeRemaining(remaining);

      if (remaining <= 0) {
        handleSubmitQuiz(true); // Auto-submit when time expires
      }
    }, 1000);

    return () => clearInterval(interval);
  }, [startTime, quizData, showResults]);

  const startNewQuiz = () => {
    setCurrentQuestionIndex(0);
    setUserAnswers(new Array(quizData.questions.length).fill(null));
    setStartTime(Date.now());
    setShowResults(false);
    setScoreData(null);
    assessmentEngine.clearProgress();
  };

  const handleAnswerSelect = (answer) => {
    const newAnswers = [...userAnswers];
    newAnswers[currentQuestionIndex] = answer;
    setUserAnswers(newAnswers);

    // Save progress
    assessmentEngine.saveProgress(
      quizData.id,
      currentQuestionIndex,
      newAnswers,
      startTime
    );
  };

  const handleNext = () => {
    if (currentQuestionIndex < quizData.questions.length - 1) {
      setCurrentQuestionIndex(currentQuestionIndex + 1);
    }
  };

  const handlePrevious = () => {
    if (currentQuestionIndex > 0) {
      setCurrentQuestionIndex(currentQuestionIndex - 1);
    }
  };

  const handleSubmitQuiz = async (autoSubmit = false) => {
    setLoading(true);

    // Check if all questions are answered
    const unansweredCount = userAnswers.filter(a => a === null || a === undefined).length;

    if (unansweredCount > 0 && !autoSubmit) {
      if (!confirm(`You have ${unansweredCount} unanswered questions. Submit anyway?`)) {
        setLoading(false);
        return;
      }
    }

    // Calculate score
    const results = assessmentEngine.calculateScore(quizData.questions, userAnswers);

    // Calculate time spent
    const timeSpent = Math.floor((Date.now() - startTime) / 1000);

    // Save results
    assessmentEngine.saveResults(quizData.id, results, timeSpent);

    // Clear progress
    assessmentEngine.clearProgress();

    setScoreData({ ...results, timeSpent });
    setShowResults(true);
    setLoading(false);

    // Notify parent component
    if (onComplete) {
      onComplete({ ...results, timeSpent });
    }
  };

  const handleRetakeQuiz = () => {
    startNewQuiz();
  };

  if (!quizData) {
    return (
      <Alert>
        <AlertTitle>No Quiz Available</AlertTitle>
        <AlertDescription>Quiz data could not be loaded.</AlertDescription>
      </Alert>
    );
  }

  // Results view
  if (showResults && scoreData) {
    const feedback = assessmentEngine.generateFeedback(scoreData.percentage, quizData.id);

    return (
      <div className="space-y-6">
        <Card>
          <CardHeader>
            <CardTitle>Quiz Results: {quizData.title}</CardTitle>
            <CardDescription>You completed the quiz!</CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            {/* Score Summary */}
            <div className="text-center p-6 bg-gradient-to-r from-blue-50 to-purple-50 rounded-lg">
              <div className="text-6xl font-bold mb-2" style={{
                color: scoreData.passed ? '#10b981' : '#ef4444'
              }}>
                {scoreData.percentage}%
              </div>
              <div className="text-xl text-gray-700">
                {scoreData.earnedPoints} / {scoreData.totalPoints} points
              </div>
              <div className="text-sm text-gray-500 mt-2">
                Time: {assessmentEngine.formatTime(scoreData.timeSpent)}
              </div>
            </div>

            {/* Feedback */}
            <Alert>
              <AlertTitle>{feedback.message}</AlertTitle>
              <AlertDescription>
                <ul className="list-disc list-inside mt-2 space-y-1">
                  {feedback.suggestions.map((suggestion, idx) => (
                    <li key={idx}>{suggestion}</li>
                  ))}
                </ul>
              </AlertDescription>
            </Alert>

            {/* Detailed Results */}
            <div>
              <h3 className="text-lg font-semibold mb-4">Question Review</h3>
              <div className="space-y-4">
                {scoreData.results.map((result, idx) => (
                  <Card key={idx} className={`border-l-4 ${
                    result.isCorrect ? 'border-l-green-500' : 'border-l-red-500'
                  }`}>
                    <CardContent className="pt-4">
                      <div className="flex items-start justify-between mb-2">
                        <div className="flex-1">
                          <p className="font-medium">Q{idx + 1}. {result.question}</p>
                        </div>
                        <div className={`px-3 py-1 rounded text-sm font-semibold ${
                          result.isCorrect
                            ? 'bg-green-100 text-green-800'
                            : 'bg-red-100 text-red-800'
                        }`}>
                          {result.isCorrect ? '✓ Correct' : '✗ Incorrect'}
                        </div>
                      </div>

                      {!result.isCorrect && (
                        <div className="mt-3 p-3 bg-yellow-50 rounded">
                          <p className="text-sm text-gray-700">
                            <span className="font-semibold">Correct answer:</span>{' '}
                            {typeof result.correctAnswer === 'number'
                              ? quizData.questions[idx].options[result.correctAnswer]
                              : result.correctAnswer}
                          </p>
                        </div>
                      )}

                      {result.explanation && (
                        <div className="mt-2 p-3 bg-blue-50 rounded">
                          <p className="text-sm text-gray-700">
                            <span className="font-semibold">Explanation:</span> {result.explanation}
                          </p>
                        </div>
                      )}
                    </CardContent>
                  </Card>
                ))}
              </div>
            </div>

            {/* Action Buttons */}
            <div className="flex gap-4">
              <Button onClick={handleRetakeQuiz} variant="outline" className="flex-1">
                Retake Quiz
              </Button>
              {onComplete && (
                <Button onClick={() => onComplete(null)} className="flex-1">
                  Continue Learning
                </Button>
              )}
            </div>
          </CardContent>
        </Card>
      </div>
    );
  }

  // Quiz in progress
  const currentQuestion = quizData.questions[currentQuestionIndex];
  const progress = ((currentQuestionIndex + 1) / quizData.questions.length) * 100;

  return (
    <div className="space-y-6">
      {/* Quiz Header */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle>{quizData.title}</CardTitle>
              <CardDescription>
                Question {currentQuestionIndex + 1} of {quizData.questions.length}
              </CardDescription>
            </div>
            {timeRemaining !== null && (
              <div className={`text-2xl font-mono font-bold ${
                timeRemaining < 60 ? 'text-red-600' : 'text-gray-700'
              }`}>
                {assessmentEngine.formatTime(timeRemaining)}
              </div>
            )}
          </div>
        </CardHeader>
        <CardContent>
          {/* Progress Bar */}
          <div className="w-full bg-gray-200 rounded-full h-2 mb-6">
            <div
              className="bg-blue-600 h-2 rounded-full transition-all duration-300"
              style={{ width: `${progress}%` }}
            />
          </div>

          {/* Question */}
          <div className="space-y-6">
            <div className="p-6 bg-gray-50 rounded-lg">
              <div className="flex items-start justify-between mb-4">
                <h3 className="text-lg font-semibold flex-1">{currentQuestion.question}</h3>
                <span className="text-sm text-gray-500 ml-4">{currentQuestion.points} pts</span>
              </div>

              {/* Answer Options */}
              {currentQuestion.type === 'multiple-choice' || currentQuestion.type === 'scenario-based' ? (
                <div className="space-y-3">
                  {currentQuestion.options.map((option, idx) => (
                    <button
                      key={idx}
                      onClick={() => handleAnswerSelect(idx)}
                      className={`w-full text-left p-4 rounded-lg border-2 transition-all ${
                        userAnswers[currentQuestionIndex] === idx
                          ? 'border-blue-600 bg-blue-50'
                          : 'border-gray-200 hover:border-blue-300 hover:bg-gray-50'
                      }`}
                    >
                      <div className="flex items-center">
                        <div className={`w-5 h-5 rounded-full border-2 mr-3 flex items-center justify-center ${
                          userAnswers[currentQuestionIndex] === idx
                            ? 'border-blue-600 bg-blue-600'
                            : 'border-gray-300'
                        }`}>
                          {userAnswers[currentQuestionIndex] === idx && (
                            <div className="w-2 h-2 bg-white rounded-full" />
                          )}
                        </div>
                        <span>{option}</span>
                      </div>
                    </button>
                  ))}
                </div>
              ) : (
                // Code completion or debugging questions
                <div>
                  {currentQuestion.placeholder && (
                    <p className="text-sm text-gray-500 mb-2">{currentQuestion.placeholder}</p>
                  )}
                  <textarea
                    value={userAnswers[currentQuestionIndex] || ''}
                    onChange={(e) => handleAnswerSelect(e.target.value)}
                    className="w-full h-32 p-3 border-2 border-gray-200 rounded-lg font-mono text-sm focus:border-blue-600 focus:outline-none"
                    placeholder="Enter your answer..."
                  />
                </div>
              )}
            </div>

            {/* Navigation Buttons */}
            <div className="flex gap-4">
              <Button
                onClick={handlePrevious}
                disabled={currentQuestionIndex === 0}
                variant="outline"
              >
                ← Previous
              </Button>

              <div className="flex-1" />

              {currentQuestionIndex < quizData.questions.length - 1 ? (
                <Button onClick={handleNext}>
                  Next →
                </Button>
              ) : (
                <Button
                  onClick={() => handleSubmitQuiz(false)}
                  disabled={loading}
                  className="bg-green-600 hover:bg-green-700"
                >
                  {loading ? 'Submitting...' : 'Submit Quiz'}
                </Button>
              )}
            </div>

            {/* Answer Status Grid */}
            <div className="pt-4 border-t">
              <p className="text-sm text-gray-600 mb-2">Question Status:</p>
              <div className="grid grid-cols-10 gap-2">
                {quizData.questions.map((_, idx) => (
                  <button
                    key={idx}
                    onClick={() => setCurrentQuestionIndex(idx)}
                    className={`h-10 rounded flex items-center justify-center text-sm font-semibold ${
                      idx === currentQuestionIndex
                        ? 'bg-blue-600 text-white'
                        : userAnswers[idx] !== null && userAnswers[idx] !== undefined
                        ? 'bg-green-100 text-green-800 border border-green-300'
                        : 'bg-gray-100 text-gray-600 border border-gray-300'
                    }`}
                  >
                    {idx + 1}
                  </button>
                ))}
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
