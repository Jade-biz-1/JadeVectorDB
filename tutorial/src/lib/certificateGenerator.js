/**
 * Certificate Generator
 *
 * Generates completion certificates for tutorial readiness assessment.
 */

/**
 * Generate certificate data
 * @param {Object} evaluation - Readiness evaluation result
 * @param {Object} userInfo - User information
 * @returns {Object} Certificate data
 */
export function generateCertificate(evaluation, userInfo = {}) {
  const certificateId = generateCertificateId();
  const issueDate = new Date();

  return {
    certificateId,
    issueDate: issueDate.toISOString(),
    issueDateFormatted: formatDate(issueDate),

    // User information
    userName: userInfo.name || 'Tutorial Participant',
    userEmail: userInfo.email || null,

    // Achievement details
    overallScore: evaluation.overallScore,
    proficiencyLevel: evaluation.proficiencyLevel.label,
    certificationLevel: getCertificationTitle(evaluation.overallScore),

    // Completion details
    completedModules: evaluation.completedModules,
    totalModules: evaluation.totalModules,
    completionPercentage: Math.round(
      (evaluation.completedModules / evaluation.totalModules) * 100
    ),

    // Skill areas
    skillAreas: Object.entries(evaluation.skillAreaScores).map(([id, area]) => ({
      name: area.name,
      score: area.score,
      status: area.status
    })),

    // Verification
    verificationUrl: `https://jadevectordb.com/verify/${certificateId}`,
    qrCodeData: generateQRCodeData(certificateId),

    // Metadata
    programName: 'JadeVectorDB Tutorial Certification',
    issuingOrganization: 'JadeVectorDB',
    validityPeriod: 'Lifetime',
    version: '1.0'
  };
}

/**
 * Generate unique certificate ID
 * @returns {string} Certificate ID
 */
function generateCertificateId() {
  const timestamp = Date.now().toString(36);
  const random = Math.random().toString(36).substring(2, 10);
  return `JVD-${timestamp}-${random}`.toUpperCase();
}

/**
 * Format date for certificate display
 * @param {Date} date - Date object
 * @returns {string} Formatted date
 */
function formatDate(date) {
  const options = { year: 'numeric', month: 'long', day: 'numeric' };
  return date.toLocaleDateString('en-US', options);
}

/**
 * Get certification title based on score
 * @param {number} score - Overall score
 * @returns {string} Certification title
 */
function getCertificationTitle(score) {
  if (score >= 95) return 'JadeVectorDB Master';
  if (score >= 85) return 'JadeVectorDB Expert';
  if (score >= 75) return 'JadeVectorDB Professional';
  if (score >= 60) return 'JadeVectorDB Associate';
  return 'JadeVectorDB Participant';
}

/**
 * Generate QR code data for verification
 * @param {string} certificateId - Certificate ID
 * @returns {string} QR code data URL
 */
function generateQRCodeData(certificateId) {
  const verificationUrl = `https://jadevectordb.com/verify/${certificateId}`;
  // In a real implementation, this would generate actual QR code
  // For now, return the URL that would be encoded
  return verificationUrl;
}

/**
 * Generate certificate HTML for download/print
 * @param {Object} certificateData - Certificate data
 * @returns {string} HTML content
 */
export function generateCertificateHTML(certificateData) {
  return `
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>JadeVectorDB Certificate - ${certificateData.userName}</title>
  <style>
    * {
      margin: 0;
      padding: 0;
      box-sizing: border-box;
    }

    body {
      font-family: 'Georgia', serif;
      background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      padding: 40px 20px;
      min-height: 100vh;
      display: flex;
      align-items: center;
      justify-center;
    }

    .certificate {
      background: white;
      max-width: 900px;
      width: 100%;
      padding: 60px;
      border: 20px solid #f8f9fa;
      box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
      position: relative;
    }

    .certificate::before {
      content: '';
      position: absolute;
      top: 10px;
      left: 10px;
      right: 10px;
      bottom: 10px;
      border: 2px solid #667eea;
      pointer-events: none;
    }

    .header {
      text-align: center;
      margin-bottom: 40px;
    }

    .logo {
      font-size: 48px;
      font-weight: bold;
      background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      background-clip: text;
      margin-bottom: 10px;
    }

    .program-name {
      font-size: 24px;
      color: #4a5568;
      margin-bottom: 20px;
    }

    .certificate-title {
      font-size: 18px;
      color: #718096;
      text-transform: uppercase;
      letter-spacing: 2px;
      margin-bottom: 30px;
    }

    .recipient {
      text-align: center;
      margin: 40px 0;
    }

    .recipient-name {
      font-size: 48px;
      color: #2d3748;
      margin-bottom: 20px;
      font-weight: normal;
      border-bottom: 2px solid #e2e8f0;
      padding-bottom: 10px;
      display: inline-block;
    }

    .achievement {
      text-align: center;
      margin: 30px 0;
      font-size: 18px;
      line-height: 1.8;
      color: #4a5568;
    }

    .achievement strong {
      color: #667eea;
      font-size: 24px;
    }

    .stats {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 20px;
      margin: 40px 0;
      padding: 30px;
      background: #f7fafc;
      border-radius: 10px;
    }

    .stat {
      text-align: center;
    }

    .stat-value {
      font-size: 36px;
      font-weight: bold;
      color: #667eea;
      margin-bottom: 5px;
    }

    .stat-label {
      font-size: 14px;
      color: #718096;
      text-transform: uppercase;
      letter-spacing: 1px;
    }

    .skill-areas {
      margin: 30px 0;
    }

    .skill-areas h3 {
      text-align: center;
      color: #2d3748;
      margin-bottom: 20px;
      font-size: 18px;
    }

    .skills-grid {
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 15px;
    }

    .skill-item {
      padding: 15px;
      background: #f7fafc;
      border-radius: 8px;
      border-left: 4px solid #667eea;
    }

    .skill-name {
      font-weight: bold;
      color: #2d3748;
      margin-bottom: 5px;
    }

    .skill-score {
      color: #667eea;
      font-size: 18px;
    }

    .footer {
      display: flex;
      justify-content: space-between;
      align-items: flex-end;
      margin-top: 50px;
      padding-top: 30px;
      border-top: 2px solid #e2e8f0;
    }

    .signature {
      text-align: center;
      flex: 1;
    }

    .signature-line {
      border-top: 2px solid #2d3748;
      padding-top: 10px;
      margin-top: 40px;
      font-weight: bold;
      color: #2d3748;
    }

    .signature-title {
      font-size: 12px;
      color: #718096;
      margin-top: 5px;
    }

    .certificate-meta {
      text-align: center;
      margin-top: 30px;
      padding: 20px;
      background: #f7fafc;
      border-radius: 8px;
    }

    .meta-item {
      display: inline-block;
      margin: 0 20px;
      font-size: 12px;
      color: #718096;
    }

    .meta-label {
      font-weight: bold;
      color: #4a5568;
    }

    @media print {
      body {
        background: white;
        padding: 0;
      }

      .certificate {
        box-shadow: none;
        max-width: 100%;
      }
    }
  </style>
</head>
<body>
  <div class="certificate">
    <div class="header">
      <div class="logo">JadeVectorDB</div>
      <div class="program-name">${certificateData.programName}</div>
      <div class="certificate-title">Certificate of Completion</div>
    </div>

    <div class="recipient">
      <div style="font-size: 16px; color: #718096; margin-bottom: 10px;">This certifies that</div>
      <div class="recipient-name">${certificateData.userName}</div>
    </div>

    <div class="achievement">
      has successfully completed the JadeVectorDB Tutorial Program<br>
      and demonstrated <strong>${certificateData.proficiencyLevel}</strong> proficiency<br>
      in vector database concepts and operations
    </div>

    <div class="stats">
      <div class="stat">
        <div class="stat-value">${certificateData.overallScore}%</div>
        <div class="stat-label">Overall Score</div>
      </div>
      <div class="stat">
        <div class="stat-value">${certificateData.completedModules}/${certificateData.totalModules}</div>
        <div class="stat-label">Modules Completed</div>
      </div>
      <div class="stat">
        <div class="stat-value">${certificateData.certificationLevel.replace('JadeVectorDB ', '')}</div>
        <div class="stat-label">Certification Level</div>
      </div>
    </div>

    <div class="skill-areas">
      <h3>Skill Areas Mastered</h3>
      <div class="skills-grid">
        ${certificateData.skillAreas.map(skill => `
          <div class="skill-item">
            <div class="skill-name">${skill.name}</div>
            <div class="skill-score">${skill.score}% - ${skill.status.toUpperCase()}</div>
          </div>
        `).join('')}
      </div>
    </div>

    <div class="footer">
      <div class="signature">
        <div class="signature-line">JadeVectorDB Team</div>
        <div class="signature-title">Tutorial Program Directors</div>
      </div>
    </div>

    <div class="certificate-meta">
      <div class="meta-item">
        <span class="meta-label">Certificate ID:</span> ${certificateData.certificateId}
      </div>
      <div class="meta-item">
        <span class="meta-label">Issue Date:</span> ${certificateData.issueDateFormatted}
      </div>
      <div class="meta-item">
        <span class="meta-label">Verify at:</span> ${certificateData.verificationUrl}
      </div>
    </div>
  </div>
</body>
</html>
  `.trim();
}

/**
 * Download certificate as HTML file
 * @param {Object} certificateData - Certificate data
 */
export function downloadCertificate(certificateData) {
  const html = generateCertificateHTML(certificateData);
  const blob = new Blob([html], { type: 'text/html' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = `JadeVectorDB-Certificate-${certificateData.certificateId}.html`;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}

/**
 * Print certificate
 * @param {Object} certificateData - Certificate data
 */
export function printCertificate(certificateData) {
  const html = generateCertificateHTML(certificateData);
  const printWindow = window.open('', '_blank');
  printWindow.document.write(html);
  printWindow.document.close();
  printWindow.focus();

  // Wait for content to load before printing
  printWindow.onload = () => {
    setTimeout(() => {
      printWindow.print();
    }, 250);
  };
}

/**
 * Share certificate (generate shareable link)
 * @param {Object} certificateData - Certificate data
 * @returns {string} Shareable URL
 */
export function generateShareableLink(certificateData) {
  // In a real implementation, this would upload to a server
  // and return a public URL
  return certificateData.verificationUrl;
}

export default {
  generateCertificate,
  generateCertificateHTML,
  downloadCertificate,
  printCertificate,
  generateShareableLink
};
