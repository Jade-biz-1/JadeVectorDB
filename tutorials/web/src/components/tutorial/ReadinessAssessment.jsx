import React, { useState, useEffect } from 'react';
import { evaluateReadiness, getProficiencyLevel, getCertificationLevel } from '../../lib/readinessEvaluation';
import { generateCertificate } from '../../lib/certificateGenerator';
import SkillsChecklist from './SkillsChecklist';
import ProductionReadinessReport from './ProductionReadinessReport';
import RecommendationsPanel from './RecommendationsPanel';
import Certificate from './Certificate';

/**
 * ReadinessAssessment - Comprehensive production readiness evaluation
 *
 * Evaluates user's overall preparedness based on all module assessments.
 */
const ReadinessAssessment = ({
  onClose,
  onContinue,
  userInfo = {}
}) => {
  const [evaluation, setEvaluation] = useState(null);
  const [certificate, setCertificate] = useState(null);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState('overview');

  // Load and evaluate readiness
  useEffect(() => {
    try {
      setLoading(true);

      // Perform readiness evaluation
      const result = evaluateReadiness();
      setEvaluation(result);

      // Generate certificate if qualified
      if (result.overallScore >= 60) {
        const cert = generateCertificate(result, userInfo);
        setCertificate(cert);
      }

      setLoading(false);
    } catch (error) {
      console.error('Failed to evaluate readiness:', error);
      setLoading(false);
    }
  }, [userInfo]);

  // Render score circle
  const renderScoreCircle = () => {
    if (!evaluation) return null;

    const { overallScore, proficiencyLevel } = evaluation;
    const circumference = 2 * Math.PI * 90;
    const strokeDashoffset = circumference - (overallScore / 100) * circumference;

    return (
      <div className="relative w-64 h-64">
        <svg className="transform -rotate-90 w-64 h-64">
          <circle
            cx="128"
            cy="128"
            r="90"
            stroke="#e5e7eb"
            strokeWidth="16"
            fill="none"
          />
          <circle
            cx="128"
            cy="128"
            r="90"
            stroke={proficiencyLevel.color}
            strokeWidth="16"
            fill="none"
            strokeDasharray={circumference}
            strokeDashoffset={strokeDashoffset}
            strokeLinecap="round"
            className="transition-all duration-1500 ease-out"
          />
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span className="text-5xl font-bold" style={{ color: proficiencyLevel.color }}>
            {overallScore}%
          </span>
          <span className="text-xl font-semibold text-gray-700 mt-2">
            {proficiencyLevel.label}
          </span>
        </div>
      </div>
    );
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-center">
          <div className="animate-spin rounded-full h-16 w-16 border-b-4 border-blue-600 mx-auto mb-4"></div>
          <p className="text-xl text-gray-600">Evaluating your readiness...</p>
          <p className="text-sm text-gray-500 mt-2">Analyzing all module assessments</p>
        </div>
      </div>
    );
  }

  if (!evaluation) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-center">
          <div className="text-red-500 text-6xl mb-4">⚠️</div>
          <h3 className="text-2xl font-bold text-gray-800 mb-2">Unable to Evaluate</h3>
          <p className="text-gray-600 mb-4">
            Please complete at least one module assessment before evaluating readiness.
          </p>
          {onClose && (
            <button
              onClick={onClose}
              className="px-6 py-3 bg-blue-600 text-white rounded-lg font-semibold hover:bg-blue-700"
            >
              Back to Tutorials
            </button>
          )}
        </div>
      </div>
    );
  }

  const { overallScore, proficiencyLevel, readyForProduction, recommendedForProduction } = evaluation;

  return (
    <div className="max-w-7xl mx-auto p-6 space-y-8">
      {/* Header */}
      <div className="text-center">
        <h1 className="text-4xl font-bold text-gray-800 mb-2">
          Production Readiness Assessment
        </h1>
        <p className="text-lg text-gray-600">
          Comprehensive evaluation of your JadeVectorDB expertise
        </p>
      </div>

      {/* Readiness Banner */}
      <div className={`p-8 rounded-lg text-center ${
        recommendedForProduction
          ? 'bg-green-50 border-2 border-green-300'
          : readyForProduction
          ? 'bg-blue-50 border-2 border-blue-300'
          : 'bg-orange-50 border-2 border-orange-300'
      }`}>
        <div className="text-6xl mb-4">
          {recommendedForProduction ? '🎉' : readyForProduction ? '👍' : '📚'}
        </div>
        <h2 className={`text-3xl font-bold mb-3 ${
          recommendedForProduction ? 'text-green-900' :
          readyForProduction ? 'text-blue-900' : 'text-orange-900'
        }`}>
          {recommendedForProduction
            ? 'Excellent! Highly Recommended for Production'
            : readyForProduction
            ? 'Good! Ready for Production with Monitoring'
            : 'Continue Learning Before Production'}
        </h2>
        <p className={`text-lg ${
          recommendedForProduction ? 'text-green-700' :
          readyForProduction ? 'text-blue-700' : 'text-orange-700'
        }`}>
          {proficiencyLevel.description}
        </p>
      </div>

      {/* Score Overview */}
      <div className="bg-white rounded-lg shadow-xl p-8">
        <div className="flex flex-col md:flex-row items-center justify-around gap-8">
          {/* Score Circle */}
          <div className="flex flex-col items-center">
            {renderScoreCircle()}
            <p className="mt-4 text-gray-600 max-w-xs text-center">
              {proficiencyLevel.recommendation}
            </p>
          </div>

          {/* Stats Grid */}
          <div className="grid grid-cols-2 gap-6">
            <div className="text-center p-6 bg-blue-50 rounded-lg">
              <div className="text-4xl font-bold text-blue-600">
                {evaluation.completedModules}/{evaluation.totalModules}
              </div>
              <div className="text-sm text-gray-600 mt-2">Modules Completed</div>
            </div>

            <div className="text-center p-6 bg-green-50 rounded-lg">
              <div className="text-4xl font-bold text-green-600">
                {Object.values(evaluation.skillAreaScores).filter(a => a.status === 'strong').length}/4
              </div>
              <div className="text-sm text-gray-600 mt-2">Strong Skill Areas</div>
            </div>

            <div className="text-center p-6 bg-purple-50 rounded-lg">
              <div className="text-4xl font-bold text-purple-600">
                {evaluation.checklist.criticalChecked}/{evaluation.checklist.criticalItems}
              </div>
              <div className="text-sm text-gray-600 mt-2">Critical Items Met</div>
            </div>

            <div className="text-center p-6 bg-yellow-50 rounded-lg">
              <div className="text-4xl font-bold text-yellow-600">
                {evaluation.skillGaps.length}
              </div>
              <div className="text-sm text-gray-600 mt-2">Skills to Improve</div>
            </div>
          </div>
        </div>
      </div>

      {/* Tabs */}
      <div className="border-b border-gray-200">
        <nav className="flex space-x-8">
          {[
            { id: 'overview', label: 'Overview', icon: '📊' },
            { id: 'skills', label: 'Skills Checklist', icon: '✓' },
            { id: 'report', label: 'Detailed Report', icon: '📋' },
            { id: 'recommendations', label: 'Recommendations', icon: '💡' },
            ...(certificate ? [{ id: 'certificate', label: 'Certificate', icon: '🏆' }] : [])
          ].map(tab => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`py-4 px-2 border-b-2 font-medium text-sm transition-colors ${
                activeTab === tab.id
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }`}
            >
              <span className="mr-2">{tab.icon}</span>
              {tab.label}
            </button>
          ))}
        </nav>
      </div>

      {/* Tab Content */}
      <div>
        {activeTab === 'overview' && (
          <div className="space-y-6">
            {/* Skill Areas Summary */}
            <div className="bg-white rounded-lg shadow p-6">
              <h3 className="text-2xl font-bold text-gray-800 mb-6">Skill Areas Performance</h3>
              <div className="space-y-4">
                {Object.entries(evaluation.skillAreaScores).map(([areaId, area]) => (
                  <div key={areaId}>
                    <div className="flex justify-between items-center mb-2">
                      <div>
                        <span className="font-semibold text-gray-800">{area.name}</span>
                        <span className="text-sm text-gray-500 ml-2">
                          ({area.skillsMet}/{area.totalSkills} skills met)
                        </span>
                      </div>
                      <div className="flex items-center gap-2">
                        <span className="font-bold" style={{
                          color: area.status === 'strong' ? '#22c55e' :
                                 area.status === 'moderate' ? '#f59e0b' : '#ef4444'
                        }}>
                          {area.score}%
                        </span>
                        <span className={`px-3 py-1 rounded-full text-xs font-semibold ${
                          area.status === 'strong' ? 'bg-green-100 text-green-800' :
                          area.status === 'moderate' ? 'bg-yellow-100 text-yellow-800' :
                          'bg-red-100 text-red-800'
                        }`}>
                          {area.status.toUpperCase()}
                        </span>
                      </div>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-3">
                      <div
                        className="h-3 rounded-full transition-all duration-500"
                        style={{
                          width: `${area.score}%`,
                          backgroundColor: area.status === 'strong' ? '#22c55e' :
                                          area.status === 'moderate' ? '#f59e0b' : '#ef4444'
                        }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Quick Stats */}
            <div className="grid md:grid-cols-2 gap-6">
              <div className="bg-white rounded-lg shadow p-6">
                <h4 className="font-bold text-gray-800 mb-4">Production Checklist</h4>
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span>Overall Completion:</span>
                    <span className="font-bold">{evaluation.checklist.percentage}%</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Critical Items:</span>
                    <span className={`font-bold ${
                      evaluation.checklist.allCriticalMet ? 'text-green-600' : 'text-red-600'
                    }`}>
                      {evaluation.checklist.criticalChecked}/{evaluation.checklist.criticalItems}
                      {evaluation.checklist.allCriticalMet && ' ✓'}
                    </span>
                  </div>
                </div>
              </div>

              <div className="bg-white rounded-lg shadow p-6">
                <h4 className="font-bold text-gray-800 mb-4">Learning Progress</h4>
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span>Modules Completed:</span>
                    <span className="font-bold">
                      {evaluation.completedModules}/{evaluation.totalModules}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span>Skills to Improve:</span>
                    <span className="font-bold text-orange-600">
                      {evaluation.skillGaps.length}
                    </span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'skills' && (
          <SkillsChecklist evaluation={evaluation} />
        )}

        {activeTab === 'report' && (
          <ProductionReadinessReport evaluation={evaluation} />
        )}

        {activeTab === 'recommendations' && (
          <RecommendationsPanel evaluation={evaluation} />
        )}

        {activeTab === 'certificate' && certificate && (
          <Certificate certificate={certificate} />
        )}
      </div>

      {/* Actions */}
      <div className="flex justify-center gap-4 pt-8 border-t">
        {onClose && (
          <button
            onClick={onClose}
            className="px-8 py-3 bg-gray-600 text-white rounded-lg font-semibold hover:bg-gray-700"
          >
            Close
          </button>
        )}
        {onContinue && readyForProduction && (
          <button
            onClick={() => onContinue(evaluation)}
            className="px-8 py-3 bg-green-600 text-white rounded-lg font-semibold hover:bg-green-700"
          >
            Continue to Production Setup
          </button>
        )}
      </div>
    </div>
  );
};

export default ReadinessAssessment;
