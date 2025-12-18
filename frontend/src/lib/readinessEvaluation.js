/**
 * Readiness Evaluation Logic
 *
 * Evaluates user's production readiness based on assessment scores
 * across all tutorial modules.
 */

import readinessCriteria from '../data/readinessCriteria.json';
import recommendations from '../data/recommendations.json';
import assessmentState from './assessmentState';

/**
 * Evaluate overall readiness based on all module assessments
 * @returns {Object} Complete readiness evaluation
 */
export function evaluateReadiness() {
  const moduleScores = getModuleScores();
  const skillAreaScores = evaluateSkillAreas(moduleScores);
  const overallScore = calculateOverallScore(skillAreaScores);
  const proficiencyLevel = getProficiencyLevel(overallScore);
  const checklist = evaluateChecklist(skillAreaScores);
  const skillGaps = identifySkillGaps(skillAreaScores);
  const personalizedRecommendations = generateRecommendations(
    proficiencyLevel,
    skillGaps
  );

  return {
    overallScore,
    proficiencyLevel,
    skillAreaScores,
    moduleScores,
    checklist,
    skillGaps,
    recommendations: personalizedRecommendations,
    readyForProduction: overallScore >= readinessCriteria.passingScore,
    recommendedForProduction: overallScore >= readinessCriteria.recommendedProductionScore,
    completedModules: Object.keys(moduleScores).length,
    totalModules: 6,
    evaluationDate: new Date().toISOString()
  };
}

/**
 * Get scores for all modules
 * @returns {Object} Module scores
 */
function getModuleScores() {
  const modules = ['module1', 'module2', 'module3', 'module4', 'module5', 'module6'];
  const scores = {};

  modules.forEach((moduleId, index) => {
    const bestScore = assessmentState.getBestScore(moduleId);
    const passed = assessmentState.hasPassedModule(moduleId);
    const stats = assessmentState.getModuleStatistics(moduleId);

    if (bestScore !== null) {
      scores[moduleId] = {
        moduleNumber: index + 1,
        bestScore,
        passed,
        attempts: stats.attempts,
        averageScore: stats.averageScore,
        improvement: stats.improvement
      };
    }
  });

  return scores;
}

/**
 * Evaluate skill areas based on module scores
 * @param {Object} moduleScores - Scores for each module
 * @returns {Object} Skill area evaluation
 */
function evaluateSkillAreas(moduleScores) {
  const skillAreas = readinessCriteria.skillAreas;
  const evaluation = {};

  Object.entries(skillAreas).forEach(([areaId, area]) => {
    const skills = area.skills;
    const skillScores = [];

    skills.forEach(skill => {
      const moduleId = skill.assessedBy;
      const moduleScore = moduleScores[moduleId];

      if (moduleScore) {
        const meetsRequirement = moduleScore.bestScore >= skill.requiredScore;
        skillScores.push({
          skillId: skill.id,
          name: skill.name,
          level: skill.level,
          requiredScore: skill.requiredScore,
          actualScore: moduleScore.bestScore,
          meetsRequirement,
          gap: meetsRequirement ? 0 : skill.requiredScore - moduleScore.bestScore
        });
      } else {
        // Module not completed
        skillScores.push({
          skillId: skill.id,
          name: skill.name,
          level: skill.level,
          requiredScore: skill.requiredScore,
          actualScore: 0,
          meetsRequirement: false,
          gap: skill.requiredScore,
          notCompleted: true
        });
      }
    });

    // Calculate area score
    const totalScore = skillScores.reduce((sum, s) => sum + s.actualScore, 0);
    const maxScore = skillScores.length * 100;
    const areaScore = (totalScore / maxScore) * 100;

    const metCount = skillScores.filter(s => s.meetsRequirement).length;
    const totalSkills = skillScores.length;

    evaluation[areaId] = {
      name: area.name,
      description: area.description,
      weight: area.weight,
      score: Math.round(areaScore),
      skillsMet: metCount,
      totalSkills,
      percentage: Math.round((metCount / totalSkills) * 100),
      skills: skillScores,
      status: getAreaStatus(areaScore)
    };
  });

  return evaluation;
}

/**
 * Get status for a skill area
 * @param {number} score - Area score (0-100)
 * @returns {string} Status (weak, moderate, strong)
 */
function getAreaStatus(score) {
  if (score >= 80) return 'strong';
  if (score >= 65) return 'moderate';
  return 'weak';
}

/**
 * Calculate weighted overall score
 * @param {Object} skillAreaScores - Scores for each skill area
 * @returns {number} Overall weighted score
 */
function calculateOverallScore(skillAreaScores) {
  let weightedSum = 0;
  let totalWeight = 0;

  Object.values(skillAreaScores).forEach(area => {
    weightedSum += area.score * area.weight;
    totalWeight += area.weight;
  });

  return Math.round(weightedSum / totalWeight);
}

/**
 * Get proficiency level based on overall score
 * @param {number} score - Overall score (0-100)
 * @returns {Object} Proficiency level information
 */
export function getProficiencyLevel(score) {
  const levels = readinessCriteria.proficiencyLevels;

  for (const [levelId, level] of Object.entries(levels)) {
    if (score >= level.min && score <= level.max) {
      return {
        id: levelId,
        ...level
      };
    }
  }

  // Default to beginner if no match
  return {
    id: 'beginner',
    ...levels.beginner
  };
}

/**
 * Evaluate production readiness checklist
 * @param {Object} skillAreaScores - Skill area scores
 * @returns {Object} Checklist evaluation
 */
function evaluateChecklist(skillAreaScores) {
  const checklistItems = readinessCriteria.productionChecklist;
  const evaluated = checklistItems.map(item => {
    const area = skillAreaScores[item.category];
    const checked = area && area.status !== 'weak';

    return {
      ...item,
      checked,
      areaScore: area ? area.score : 0
    };
  });

  const checkedCount = evaluated.filter(item => item.checked).length;
  const criticalCount = evaluated.filter(item => item.critical).length;
  const criticalChecked = evaluated.filter(item => item.critical && item.checked).length;

  return {
    items: evaluated,
    totalItems: evaluated.length,
    checkedItems: checkedCount,
    percentage: Math.round((checkedCount / evaluated.length) * 100),
    criticalItems: criticalCount,
    criticalChecked,
    allCriticalMet: criticalChecked === criticalCount
  };
}

/**
 * Identify skill gaps
 * @param {Object} skillAreaScores - Skill area scores
 * @returns {Array} List of skill gaps
 */
function identifySkillGaps(skillAreaScores) {
  const gaps = [];

  Object.entries(skillAreaScores).forEach(([areaId, area]) => {
    area.skills.forEach(skill => {
      if (!skill.meetsRequirement) {
        gaps.push({
          area: areaId,
          areaName: area.name,
          skillName: skill.name,
          level: skill.level,
          gap: skill.gap,
          requiredScore: skill.requiredScore,
          actualScore: skill.actualScore,
          priority: skill.level === 'advanced' ? 'high' : 'medium',
          notCompleted: skill.notCompleted || false
        });
      }
    });
  });

  // Sort by gap size (largest first)
  gaps.sort((a, b) => b.gap - a.gap);

  return gaps;
}

/**
 * Generate personalized recommendations
 * @param {Object} proficiencyLevel - User's proficiency level
 * @param {Array} skillGaps - Identified skill gaps
 * @returns {Object} Personalized recommendations
 */
function generateRecommendations(proficiencyLevel, skillGaps) {
  const levelRecs = recommendations.byProficiencyLevel[proficiencyLevel.id] ||
                    recommendations.byProficiencyLevel.beginner;

  // Get skill area specific recommendations
  const areaRecs = {};
  const uniqueAreas = [...new Set(skillGaps.map(gap => gap.area))];

  uniqueAreas.forEach(areaId => {
    const areaGaps = skillGaps.filter(gap => gap.area === areaId);
    const gapLevel = areaGaps.length > 3 ? 'weak' :
                     areaGaps.length > 1 ? 'moderate' : 'strong';

    areaRecs[areaId] = recommendations.bySkillArea[areaId][gapLevel];
  });

  // Get relevant practice projects
  const suggestedProjects = recommendations.practiceProjects.filter(project => {
    if (proficiencyLevel.id === 'beginner') {
      return project.difficulty === 'beginner';
    } else if (proficiencyLevel.id === 'intermediate') {
      return project.difficulty === 'beginner' || project.difficulty === 'intermediate';
    } else {
      return true; // All projects for advanced users
    }
  });

  return {
    nextSteps: levelRecs.nextSteps,
    learningResources: levelRecs.learningResources,
    skillAreaRecommendations: areaRecs,
    practiceProjects: suggestedProjects.slice(0, 3), // Top 3
    commonChallenges: recommendations.commonChallenges,
    estimatedTimeToProduction: levelRecs.estimatedTimeToProduction
  };
}

/**
 * Get certification level based on score
 * @param {number} score - Overall score
 * @returns {Object} Certification information
 */
export function getCertificationLevel(score) {
  const certPath = recommendations.certificationPath;

  if (score >= 95) return { level: 'platinum', ...certPath.platinum };
  if (score >= 85) return { level: 'gold', ...certPath.gold };
  if (score >= 75) return { level: 'silver', ...certPath.silver };
  if (score >= 60) return { level: 'bronze', ...certPath.bronze };

  return null; // No certification
}

/**
 * Compare readiness over time
 * @param {Object} previousEvaluation - Previous evaluation result
 * @param {Object} currentEvaluation - Current evaluation result
 * @returns {Object} Comparison result
 */
export function compareReadiness(previousEvaluation, currentEvaluation) {
  return {
    scoreChange: currentEvaluation.overallScore - previousEvaluation.overallScore,
    skillImprovements: identifyImprovements(
      previousEvaluation.skillAreaScores,
      currentEvaluation.skillAreaScores
    ),
    gapsResolved: previousEvaluation.skillGaps.length - currentEvaluation.skillGaps.length,
    newGaps: identifyNewGaps(
      previousEvaluation.skillGaps,
      currentEvaluation.skillGaps
    ),
    levelChange: {
      from: previousEvaluation.proficiencyLevel.id,
      to: currentEvaluation.proficiencyLevel.id,
      improved: getProficiencyRank(currentEvaluation.proficiencyLevel.id) >
               getProficiencyRank(previousEvaluation.proficiencyLevel.id)
    }
  };
}

/**
 * Identify skill improvements
 * @param {Object} previousScores - Previous skill area scores
 * @param {Object} currentScores - Current skill area scores
 * @returns {Array} List of improvements
 */
function identifyImprovements(previousScores, currentScores) {
  const improvements = [];

  Object.keys(currentScores).forEach(areaId => {
    const prev = previousScores[areaId];
    const curr = currentScores[areaId];

    if (prev && curr.score > prev.score) {
      improvements.push({
        area: areaId,
        areaName: curr.name,
        improvement: curr.score - prev.score,
        from: prev.score,
        to: curr.score
      });
    }
  });

  return improvements.sort((a, b) => b.improvement - a.improvement);
}

/**
 * Identify new skill gaps
 * @param {Array} previousGaps - Previous gaps
 * @param {Array} currentGaps - Current gaps
 * @returns {Array} New gaps
 */
function identifyNewGaps(previousGaps, currentGaps) {
  const prevGapIds = new Set(previousGaps.map(g => g.skillName));
  return currentGaps.filter(gap => !prevGapIds.has(gap.skillName));
}

/**
 * Get numeric rank for proficiency level
 * @param {string} levelId - Level identifier
 * @returns {number} Rank (higher is better)
 */
function getProficiencyRank(levelId) {
  const ranks = {
    beginner: 1,
    intermediate: 2,
    proficient: 3,
    expert: 4,
    master: 5
  };
  return ranks[levelId] || 0;
}

/**
 * Export readiness data for sharing/printing
 * @param {Object} evaluation - Readiness evaluation
 * @returns {Object} Exportable data
 */
export function exportReadinessData(evaluation) {
  return {
    ...evaluation,
    exportDate: new Date().toISOString(),
    version: '1.0'
  };
}

export default {
  evaluateReadiness,
  getProficiencyLevel,
  getCertificationLevel,
  compareReadiness,
  exportReadinessData
};
