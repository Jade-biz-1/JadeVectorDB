import React, { useState } from 'react';

/**
 * SkillsChecklist - Interactive checklist of production readiness criteria
 */
const SkillsChecklist = ({ evaluation }) => {
  const [expandedAreas, setExpandedAreas] = useState(new Set());

  const toggleArea = (areaId) => {
    const newExpanded = new Set(expandedAreas);
    if (newExpanded.has(areaId)) {
      newExpanded.delete(areaId);
    } else {
      newExpanded.add(areaId);
    }
    setExpandedAreas(newExpanded);
  };

  return (
    <div className="space-y-6">
      {/* Production Checklist */}
      <div className="bg-white rounded-lg shadow p-6">
        <div className="flex items-center justify-between mb-6">
          <h3 className="text-2xl font-bold text-gray-800">
            Production Readiness Checklist
          </h3>
          <div className="text-right">
            <div className="text-3xl font-bold text-blue-600">
              {evaluation.checklist.percentage}%
            </div>
            <div className="text-sm text-gray-600">
              {evaluation.checklist.checkedItems}/{evaluation.checklist.totalItems} items
            </div>
          </div>
        </div>

        <div className="space-y-2">
          {evaluation.checklist.items.map(item => (
            <div
              key={item.id}
              className={`p-4 rounded-lg border-2 ${
                item.checked
                  ? 'bg-green-50 border-green-300'
                  : 'bg-gray-50 border-gray-300'
              }`}
            >
              <div className="flex items-start gap-3">
                <div className={`flex-shrink-0 w-6 h-6 rounded border-2 flex items-center justify-center ${
                  item.checked
                    ? 'bg-green-500 border-green-500'
                    : 'bg-white border-gray-400'
                }`}>
                  {item.checked && (
                    <span className="text-white text-sm">✓</span>
                  )}
                </div>
                <div className="flex-1">
                  <div className="flex items-center gap-2">
                    <span className={`${item.checked ? 'text-green-900' : 'text-gray-700'} font-medium`}>
                      {item.item}
                    </span>
                    {item.critical && (
                      <span className="px-2 py-1 bg-red-100 text-red-800 text-xs font-semibold rounded">
                        CRITICAL
                      </span>
                    )}
                  </div>
                  <div className="text-sm text-gray-600 mt-1">
                    Area Score: {item.areaScore}%
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>

        {!evaluation.checklist.allCriticalMet && (
          <div className="mt-6 p-4 bg-red-50 border border-red-200 rounded-lg">
            <p className="text-red-800 font-medium">
              ⚠️ Not all critical items are met. Focus on critical areas before production deployment.
            </p>
          </div>
        )}
      </div>

      {/* Detailed Skill Areas */}
      <div className="bg-white rounded-lg shadow p-6">
        <h3 className="text-2xl font-bold text-gray-800 mb-6">
          Detailed Skills Assessment
        </h3>

        <div className="space-y-4">
          {Object.entries(evaluation.skillAreaScores).map(([areaId, area]) => (
            <div key={areaId} className="border rounded-lg overflow-hidden">
              {/* Area Header */}
              <button
                onClick={() => toggleArea(areaId)}
                className="w-full p-4 bg-gray-50 hover:bg-gray-100 transition-colors flex items-center justify-between"
              >
                <div className="flex items-center gap-3">
                  <span className="text-2xl">
                    {expandedAreas.has(areaId) ? '▼' : '▶'}
                  </span>
                  <div className="text-left">
                    <div className="font-bold text-gray-800">{area.name}</div>
                    <div className="text-sm text-gray-600">{area.description}</div>
                  </div>
                </div>
                <div className="flex items-center gap-4">
                  <div className="text-right">
                    <div className="font-bold text-2xl" style={{
                      color: area.status === 'strong' ? '#22c55e' :
                             area.status === 'moderate' ? '#f59e0b' : '#ef4444'
                    }}>
                      {area.score}%
                    </div>
                    <div className="text-xs text-gray-600">
                      {area.skillsMet}/{area.totalSkills} skills
                    </div>
                  </div>
                  <span className={`px-3 py-1 rounded-full text-xs font-semibold ${
                    area.status === 'strong' ? 'bg-green-100 text-green-800' :
                    area.status === 'moderate' ? 'bg-yellow-100 text-yellow-800' :
                    'bg-red-100 text-red-800'
                  }`}>
                    {area.status.toUpperCase()}
                  </span>
                </div>
              </button>

              {/* Expanded Skills */}
              {expandedAreas.has(areaId) && (
                <div className="p-4 bg-white space-y-3">
                  {area.skills.map(skill => (
                    <div
                      key={skill.skillId}
                      className={`p-3 rounded-lg border ${
                        skill.meetsRequirement
                          ? 'bg-green-50 border-green-200'
                          : skill.notCompleted
                          ? 'bg-gray-50 border-gray-200'
                          : 'bg-orange-50 border-orange-200'
                      }`}
                    >
                      <div className="flex items-start justify-between">
                        <div className="flex-1">
                          <div className="font-medium text-gray-800">{skill.name}</div>
                          <div className="text-sm text-gray-600 mt-1">
                            Level: {skill.level} • Required: {skill.requiredScore}%
                          </div>
                        </div>
                        <div className="text-right">
                          {skill.notCompleted ? (
                            <div className="text-gray-500 font-medium">Not Completed</div>
                          ) : (
                            <>
                              <div className={`text-2xl font-bold ${
                                skill.meetsRequirement ? 'text-green-600' : 'text-orange-600'
                              }`}>
                                {skill.actualScore}%
                              </div>
                              {skill.meetsRequirement ? (
                                <div className="text-green-600 text-sm">✓ Met</div>
                              ) : (
                                <div className="text-orange-600 text-sm">
                                  Gap: {skill.gap}%
                                </div>
                              )}
                            </>
                          )}
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default SkillsChecklist;
