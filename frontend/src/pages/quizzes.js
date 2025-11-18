import Head from 'next/head';
import { useState } from 'react';
import { Button } from '../components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../components/ui/card';
import { Alert, AlertTitle, AlertDescription } from '../components/ui/alert';
import Quiz from '../components/Quiz';
import { quizQuestions, getQuizTitles } from '../data/quizQuestions';
import assessmentEngine from '../lib/assessmentEngine';

export default function Quizzes() {
  const [selectedQuiz, setSelectedQuiz] = useState(null);
  const [showQuiz, setShowQuiz] = useState(false);

  const quizList = getQuizTitles();
  const stats = assessmentEngine.getStatistics();

  const handleStartQuiz = (quizId) => {
    setSelectedQuiz(quizQuestions[quizId]);
    setShowQuiz(true);
  };

  const handleQuizComplete = (results) => {
    if (!results) {
      // User wants to continue learning
      setShowQuiz(false);
      setSelectedQuiz(null);
    }
  };

  const handleBackToList = () => {
    setShowQuiz(false);
    setSelectedQuiz(null);
  };

  const getModuleStats = (quizId) => {
    return stats.moduleStats?.[quizId] || null;
  };

  if (showQuiz && selectedQuiz) {
    return (
      <>
        <Head>
          <title>{selectedQuiz.title} - JadeVectorDB</title>
          <meta name="description" content={selectedQuiz.description} />
        </Head>

        <div className="min-h-screen bg-gradient-to-br from-blue-50 to-purple-50 p-8">
          <div className="max-w-4xl mx-auto">
            <div className="mb-6">
              <Button onClick={handleBackToList} variant="outline">
                ← Back to Quiz List
              </Button>
            </div>

            <Quiz quizData={selectedQuiz} onComplete={handleQuizComplete} />
          </div>
        </div>
      </>
    );
  }

  return (
    <>
      <Head>
        <title>Quizzes - JadeVectorDB Tutorial Assessment</title>
        <meta name="description" content="Test your knowledge with JadeVectorDB tutorial quizzes" />
      </Head>

      <div className="min-h-screen bg-gradient-to-br from-blue-50 to-purple-50 p-8">
        <div className="max-w-6xl mx-auto">
          {/* Header */}
          <div className="mb-8">
            <h1 className="text-4xl font-bold mb-2 bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
              Tutorial Quizzes
            </h1>
            <p className="text-gray-600 text-lg">
              Test your knowledge and track your learning progress
            </p>
          </div>

          {/* Overall Statistics */}
          {stats.totalQuizzes > 0 && (
            <Card className="mb-8">
              <CardHeader>
                <CardTitle>Your Progress</CardTitle>
                <CardDescription>Overall quiz statistics</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
                  <div className="text-center">
                    <div className="text-3xl font-bold text-blue-600">{stats.totalQuizzes}</div>
                    <div className="text-sm text-gray-600">Total Attempts</div>
                  </div>
                  <div className="text-center">
                    <div className="text-3xl font-bold text-green-600">{stats.passedQuizzes}</div>
                    <div className="text-sm text-gray-600">Passed</div>
                  </div>
                  <div className="text-center">
                    <div className="text-3xl font-bold text-purple-600">
                      {Math.round(stats.averageScore)}%
                    </div>
                    <div className="text-sm text-gray-600">Average Score</div>
                  </div>
                  <div className="text-center">
                    <div className="text-3xl font-bold text-orange-600">
                      {assessmentEngine.formatTime(stats.totalTimeSpent)}
                    </div>
                    <div className="text-sm text-gray-600">Total Time</div>
                  </div>
                </div>
              </CardContent>
            </Card>
          )}

          {/* Introduction Alert */}
          <Alert className="mb-8">
            <AlertTitle>📚 How to Use Quizzes</AlertTitle>
            <AlertDescription>
              <ul className="list-disc list-inside mt-2 space-y-1">
                <li>Each quiz contains multiple question types: multiple-choice, code completion, and debugging</li>
                <li>You need 70% or higher to pass</li>
                <li>Your progress is automatically saved - you can resume anytime</li>
                <li>Review your answers and explanations after completing the quiz</li>
                <li>Retake quizzes as many times as you like to improve your score</li>
              </ul>
            </AlertDescription>
          </Alert>

          {/* Quiz Cards */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {quizList.map((quiz) => {
              const quizData = quizQuestions[quiz.id];
              const moduleStats = getModuleStats(quiz.id);
              const bestScore = moduleStats?.bestScore || 0;
              const passed = moduleStats?.passed || false;

              return (
                <Card
                  key={quiz.id}
                  className={`hover:shadow-lg transition-shadow ${
                    passed ? 'border-l-4 border-l-green-500' : ''
                  }`}
                >
                  <CardHeader>
                    <div className="flex items-start justify-between">
                      <div className="flex-1">
                        <CardTitle className="flex items-center gap-2">
                          {quiz.title}
                          {passed && <span className="text-green-600">✓</span>}
                        </CardTitle>
                        <CardDescription>{quiz.description}</CardDescription>
                      </div>
                      {moduleStats && (
                        <div className="text-right ml-4">
                          <div className={`text-2xl font-bold ${
                            bestScore >= 90 ? 'text-green-600' :
                            bestScore >= 70 ? 'text-blue-600' :
                            bestScore >= 50 ? 'text-orange-600' :
                            'text-red-600'
                          }`}>
                            {bestScore}%
                          </div>
                          <div className="text-xs text-gray-500">Best Score</div>
                        </div>
                      )}
                    </div>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-4">
                      {/* Quiz Info */}
                      <div className="flex items-center gap-6 text-sm text-gray-600">
                        <div className="flex items-center gap-1">
                          <span className="font-semibold">{quizData.questions.length}</span>
                          <span>Questions</span>
                        </div>
                        <div className="flex items-center gap-1">
                          <span className="font-semibold">
                            {assessmentEngine.formatTime(quizData.timeLimit)}
                          </span>
                          <span>Time Limit</span>
                        </div>
                        <div className="flex items-center gap-1">
                          <span className="font-semibold">{quizData.passingScore}%</span>
                          <span>to Pass</span>
                        </div>
                      </div>

                      {/* Module Stats */}
                      {moduleStats && moduleStats.attempts > 0 && (
                        <div className="pt-4 border-t">
                          <div className="grid grid-cols-3 gap-4 text-sm">
                            <div>
                              <div className="text-gray-600">Attempts</div>
                              <div className="font-semibold">{moduleStats.attempts}</div>
                            </div>
                            <div>
                              <div className="text-gray-600">Average</div>
                              <div className="font-semibold">{moduleStats.averageScore}%</div>
                            </div>
                            <div>
                              <div className="text-gray-600">Time Spent</div>
                              <div className="font-semibold">
                                {assessmentEngine.formatTime(moduleStats.totalTimeSpent)}
                              </div>
                            </div>
                          </div>
                        </div>
                      )}

                      {/* Action Button */}
                      <Button
                        onClick={() => handleStartQuiz(quiz.id)}
                        className="w-full"
                        variant={passed ? "outline" : "default"}
                      >
                        {moduleStats && moduleStats.attempts > 0
                          ? passed
                            ? 'Retake Quiz'
                            : 'Try Again'
                          : 'Start Quiz'}
                      </Button>
                    </div>
                  </CardContent>
                </Card>
              );
            })}
          </div>

          {/* Empty State */}
          {quizList.length === 0 && (
            <Card>
              <CardContent className="text-center py-12">
                <p className="text-gray-500">No quizzes available yet.</p>
              </CardContent>
            </Card>
          )}

          {/* Learning Tips */}
          <Card className="mt-8">
            <CardHeader>
              <CardTitle>💡 Learning Tips</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <h4 className="font-semibold mb-2">Before Taking a Quiz:</h4>
                  <ul className="list-disc list-inside text-sm text-gray-700 space-y-1">
                    <li>Complete the corresponding tutorial module</li>
                    <li>Practice with the CLI examples</li>
                    <li>Review the documentation</li>
                    <li>Try building a small project</li>
                  </ul>
                </div>
                <div>
                  <h4 className="font-semibold mb-2">After Failing a Quiz:</h4>
                  <ul className="list-disc list-inside text-sm text-gray-700 space-y-1">
                    <li>Review the questions you got wrong</li>
                    <li>Read the explanations carefully</li>
                    <li>Revisit the tutorial content</li>
                    <li>Practice more before retaking</li>
                  </ul>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Export Results */}
          {stats.totalQuizzes > 0 && (
            <div className="mt-8 text-center">
              <Button
                onClick={() => {
                  const exportData = assessmentEngine.exportResults();
                  const blob = new Blob([JSON.stringify(exportData, null, 2)], {
                    type: 'application/json'
                  });
                  const url = URL.createObjectURL(blob);
                  const a = document.createElement('a');
                  a.href = url;
                  a.download = `jadevectordb-quiz-results-${new Date().toISOString()}.json`;
                  a.click();
                  URL.revokeObjectURL(url);
                }}
                variant="outline"
              >
                📊 Export All Results
              </Button>
            </div>
          )}
        </div>
      </div>
    </>
  );
}
