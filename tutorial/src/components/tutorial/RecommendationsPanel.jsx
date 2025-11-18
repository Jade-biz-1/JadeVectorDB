import React, { useState } from 'react';

/**
 * RecommendationsPanel - Personalized learning recommendations
 */
const RecommendationsPanel = ({ evaluation }) => {
  const [expandedSections, setExpandedSections] = useState(new Set(['nextSteps']));

  const toggleSection = (section) => {
    const newExpanded = new Set(expandedSections);
    if (newExpanded.has(section)) {
      newExpanded.delete(section);
    } else {
      newExpanded.add(section);
    }
    setExpandedSections(newExpanded);
  };

  const { recommendations } = evaluation;

  return (
    <div className="space-y-6">
      {/* Next Steps */}
      <div className="bg-white rounded-lg shadow overflow-hidden">
        <button
          onClick={() => toggleSection('nextSteps')}
          className="w-full p-6 bg-blue-50 hover:bg-blue-100 transition-colors flex items-center justify-between"
        >
          <h3 className="text-2xl font-bold text-blue-900 flex items-center gap-2">
            <span>🎯</span> Recommended Next Steps
          </h3>
          <span className="text-2xl text-blue-900">
            {expandedSections.has('nextSteps') ? '▼' : '▶'}
          </span>
        </button>
        {expandedSections.has('nextSteps') && (
          <div className="p-6">
            <ol className="space-y-3 list-decimal list-inside">
              {recommendations.nextSteps.map((step, index) => (
                <li key={index} className="text-gray-700 leading-relaxed">
                  {step}
                </li>
              ))}
            </ol>
            {recommendations.estimatedTimeToProduction && (
              <div className="mt-6 p-4 bg-blue-50 rounded-lg">
                <p className="text-blue-900 font-medium">
                  ⏱️ Estimated time to production: {recommendations.estimatedTimeToProduction}
                </p>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Learning Resources */}
      <div className="bg-white rounded-lg shadow overflow-hidden">
        <button
          onClick={() => toggleSection('resources')}
          className="w-full p-6 bg-green-50 hover:bg-green-100 transition-colors flex items-center justify-between"
        >
          <h3 className="text-2xl font-bold text-green-900 flex items-center gap-2">
            <span>📚</span> Learning Resources
          </h3>
          <span className="text-2xl text-green-900">
            {expandedSections.has('resources') ? '▼' : '▶'}
          </span>
        </button>
        {expandedSections.has('resources') && (
          <div className="p-6 space-y-4">
            {recommendations.learningResources.map((resource, index) => (
              <div
                key={index}
                className={`p-4 rounded-lg border-2 ${
                  resource.priority === 'high'
                    ? 'bg-red-50 border-red-300'
                    : resource.priority === 'medium'
                    ? 'bg-yellow-50 border-yellow-300'
                    : 'bg-blue-50 border-blue-300'
                }`}
              >
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <h4 className="font-bold text-gray-800">{resource.title}</h4>
                    <p className="text-sm text-gray-600 mt-1">{resource.description}</p>
                  </div>
                  <span className={`px-3 py-1 rounded-full text-xs font-semibold ${
                    resource.priority === 'high'
                      ? 'bg-red-100 text-red-800'
                      : resource.priority === 'medium'
                      ? 'bg-yellow-100 text-yellow-800'
                      : 'bg-blue-100 text-blue-800'
                  }`}>
                    {resource.priority.toUpperCase()}
                  </span>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Skill Area Recommendations */}
      {Object.keys(recommendations.skillAreaRecommendations).length > 0 && (
        <div className="bg-white rounded-lg shadow overflow-hidden">
          <button
            onClick={() => toggleSection('skillAreas')}
            className="w-full p-6 bg-purple-50 hover:bg-purple-100 transition-colors flex items-center justify-between"
          >
            <h3 className="text-2xl font-bold text-purple-900 flex items-center gap-2">
              <span>🎓</span> Skill Area Focus
            </h3>
            <span className="text-2xl text-purple-900">
              {expandedSections.has('skillAreas') ? '▼' : '▶'}
            </span>
          </button>
          {expandedSections.has('skillAreas') && (
            <div className="p-6 space-y-4">
              {Object.entries(recommendations.skillAreaRecommendations).map(([areaId, rec]) => (
                <div key={areaId} className="p-4 bg-purple-50 rounded-lg border border-purple-200">
                  <p className="text-purple-900 font-medium mb-2">{rec.recommendation}</p>
                  {rec.resources.length > 0 && (
                    <div className="mt-3">
                      <p className="text-sm text-purple-800 font-semibold mb-2">Recommended Resources:</p>
                      <ul className="list-disc list-inside text-sm text-purple-700 space-y-1">
                        {rec.resources.map((resource, index) => (
                          <li key={index}>{resource}</li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Practice Projects */}
      {recommendations.practiceProjects && recommendations.practiceProjects.length > 0 && (
        <div className="bg-white rounded-lg shadow overflow-hidden">
          <button
            onClick={() => toggleSection('projects')}
            className="w-full p-6 bg-yellow-50 hover:bg-yellow-100 transition-colors flex items-center justify-between"
          >
            <h3 className="text-2xl font-bold text-yellow-900 flex items-center gap-2">
              <span>🛠️</span> Practice Projects
            </h3>
            <span className="text-2xl text-yellow-900">
              {expandedSections.has('projects') ? '▼' : '▶'}
            </span>
          </button>
          {expandedSections.has('projects') && (
            <div className="p-6 space-y-4">
              {recommendations.practiceProjects.map((project, index) => (
                <div key={index} className="p-4 bg-yellow-50 rounded-lg border border-yellow-200">
                  <div className="flex items-start justify-between mb-2">
                    <h4 className="font-bold text-gray-800">{project.title}</h4>
                    <span className={`px-3 py-1 rounded-full text-xs font-semibold ${
                      project.difficulty === 'beginner'
                        ? 'bg-green-100 text-green-800'
                        : project.difficulty === 'intermediate'
                        ? 'bg-yellow-100 text-yellow-800'
                        : 'bg-red-100 text-red-800'
                    }`}>
                      {project.difficulty.toUpperCase()}
                    </span>
                  </div>
                  <p className="text-sm text-gray-700 mb-2">{project.description}</p>
                  <div className="flex items-center gap-4 text-sm text-gray-600">
                    <span>⏱️ {project.estimatedTime}</span>
                    <span>📋 {project.skills.join(', ')}</span>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Common Challenges */}
      {recommendations.commonChallenges && recommendations.commonChallenges.length > 0 && (
        <div className="bg-white rounded-lg shadow overflow-hidden">
          <button
            onClick={() => toggleSection('challenges')}
            className="w-full p-6 bg-red-50 hover:bg-red-100 transition-colors flex items-center justify-between"
          >
            <h3 className="text-2xl font-bold text-red-900 flex items-center gap-2">
              <span>⚠️</span> Common Challenges & Solutions
            </h3>
            <span className="text-2xl text-red-900">
              {expandedSections.has('challenges') ? '▼' : '▶'}
            </span>
          </button>
          {expandedSections.has('challenges') && (
            <div className="p-6 space-y-4">
              {recommendations.commonChallenges.map((item, index) => (
                <div key={index} className="p-4 bg-red-50 rounded-lg border border-red-200">
                  <h4 className="font-bold text-red-900 mb-2">Challenge: {item.challenge}</h4>
                  <p className="text-sm text-red-800 mb-2">
                    <strong>Solution:</strong> {item.solution}
                  </p>
                  {item.resources && item.resources.length > 0 && (
                    <div className="text-sm text-red-700">
                      <strong>Resources:</strong> {item.resources.join(', ')}
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default RecommendationsPanel;
