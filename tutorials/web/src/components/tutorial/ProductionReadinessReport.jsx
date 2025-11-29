import React from 'react';

/**
 * ProductionReadinessReport - Detailed production readiness report
 */
const ProductionReadinessReport = ({ evaluation }) => {
  return (
    <div className="space-y-6">
      {/* Executive Summary */}
      <div className="bg-white rounded-lg shadow p-6">
        <h3 className="text-2xl font-bold text-gray-800 mb-4">Executive Summary</h3>
        <div className="prose max-w-none">
          <p className="text-gray-700 leading-relaxed">
            Based on your performance across all tutorial modules, you have achieved a{' '}
            <strong className="text-blue-600">{evaluation.proficiencyLevel.label}</strong> level
            proficiency with an overall score of <strong>{evaluation.overallScore}%</strong>.
          </p>
          <p className="text-gray-700 leading-relaxed mt-4">
            {evaluation.recommendedForProduction
              ? 'You have demonstrated excellent understanding and are highly recommended for production deployment.'
              : evaluation.readyForProduction
              ? 'You have sufficient knowledge for production deployment with proper monitoring and support.'
              : 'Additional study and practice are recommended before production deployment.'}
          </p>
        </div>
      </div>

      {/* Strengths */}
      <div className="bg-white rounded-lg shadow p-6">
        <h3 className="text-2xl font-bold text-gray-800 mb-4 flex items-center gap-2">
          <span>💪</span> Strengths
        </h3>
        <div className="space-y-3">
          {Object.entries(evaluation.skillAreaScores)
            .filter(([_, area]) => area.status === 'strong')
            .map(([areaId, area]) => (
              <div key={areaId} className="p-4 bg-green-50 border border-green-200 rounded-lg">
                <div className="flex items-center justify-between">
                  <div>
                    <div className="font-semibold text-green-900">{area.name}</div>
                    <div className="text-sm text-green-700 mt-1">
                      {area.skillsMet}/{area.totalSkills} skills mastered
                    </div>
                  </div>
                  <div className="text-3xl font-bold text-green-600">{area.score}%</div>
                </div>
              </div>
            ))}
          {Object.values(evaluation.skillAreaScores).every(a => a.status !== 'strong') && (
            <p className="text-gray-600 italic">
              Keep learning to build strong competencies in all skill areas.
            </p>
          )}
        </div>
      </div>

      {/* Areas for Improvement */}
      {evaluation.skillGaps.length > 0 && (
        <div className="bg-white rounded-lg shadow p-6">
          <h3 className="text-2xl font-bold text-gray-800 mb-4 flex items-center gap-2">
            <span>📈</span> Areas for Improvement
          </h3>
          <div className="space-y-3">
            {evaluation.skillGaps.slice(0, 5).map((gap, index) => (
              <div key={index} className="p-4 bg-orange-50 border border-orange-200 rounded-lg">
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <div className="font-semibold text-orange-900">{gap.skillName}</div>
                    <div className="text-sm text-orange-700 mt-1">
                      Area: {gap.areaName} • Level: {gap.level}
                    </div>
                  </div>
                  <div className="text-right">
                    {gap.notCompleted ? (
                      <div className="text-gray-600 font-medium">Not Completed</div>
                    ) : (
                      <>
                        <div className="text-orange-600 font-bold">Gap: {gap.gap}%</div>
                        <div className="text-sm text-gray-600">
                          {gap.actualScore}% / {gap.requiredScore}%
                        </div>
                      </>
                    )}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Module Scores */}
      <div className="bg-white rounded-lg shadow p-6">
        <h3 className="text-2xl font-bold text-gray-800 mb-4">Module Performance</h3>
        <div className="space-y-3">
          {Object.entries(evaluation.moduleScores).map(([moduleId, module]) => (
            <div key={moduleId} className="p-4 bg-gray-50 rounded-lg">
              <div className="flex items-center justify-between mb-2">
                <div>
                  <div className="font-semibold text-gray-800">
                    Module {module.moduleNumber}
                  </div>
                  <div className="text-sm text-gray-600">
                    {module.attempts} attempt(s) • Avg: {module.averageScore}%
                  </div>
                </div>
                <div className="text-right">
                  <div className="text-2xl font-bold text-blue-600">
                    {module.bestScore}%
                  </div>
                  {module.passed && (
                    <div className="text-green-600 text-sm font-semibold">✓ Passed</div>
                  )}
                </div>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div
                  className={`h-2 rounded-full ${
                    module.passed ? 'bg-green-500' : 'bg-orange-500'
                  }`}
                  style={{ width: `${module.bestScore}%` }}
                />
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Production Deployment Readiness */}
      <div className={`rounded-lg shadow p-6 ${
        evaluation.checklist.allCriticalMet
          ? 'bg-green-50 border-2 border-green-300'
          : 'bg-orange-50 border-2 border-orange-300'
      }`}>
        <h3 className="text-2xl font-bold mb-4 ${
          evaluation.checklist.allCriticalMet ? 'text-green-900' : 'text-orange-900'
        }">
          Production Deployment Status
        </h3>
        <div className="grid md:grid-cols-2 gap-4">
          <div className="p-4 bg-white rounded-lg">
            <div className="text-sm text-gray-600 mb-1">Overall Readiness</div>
            <div className="text-3xl font-bold text-blue-600">
              {evaluation.overallScore}%
            </div>
          </div>
          <div className="p-4 bg-white rounded-lg">
            <div className="text-sm text-gray-600 mb-1">Critical Items</div>
            <div className={`text-3xl font-bold ${
              evaluation.checklist.allCriticalMet ? 'text-green-600' : 'text-orange-600'
            }`}>
              {evaluation.checklist.criticalChecked}/{evaluation.checklist.criticalItems}
            </div>
          </div>
        </div>
        <div className="mt-4 p-4 bg-white rounded-lg">
          <p className={`font-medium ${
            evaluation.checklist.allCriticalMet ? 'text-green-800' : 'text-orange-800'
          }`}>
            {evaluation.checklist.allCriticalMet
              ? '✓ All critical production requirements are met. You are ready to proceed with deployment planning.'
              : '⚠ Some critical requirements are not yet met. Review the checklist and address gaps before production deployment.'}
          </p>
        </div>
      </div>

      {/* Evaluation Metadata */}
      <div className="bg-gray-50 rounded-lg p-4 text-sm text-gray-600 text-center">
        <p>
          Evaluation Date: {new Date(evaluation.evaluationDate).toLocaleDateString()} •
          Completed Modules: {evaluation.completedModules}/{evaluation.totalModules} •
          Version: 1.0
        </p>
      </div>
    </div>
  );
};

export default ProductionReadinessReport;
