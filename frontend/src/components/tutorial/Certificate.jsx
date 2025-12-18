import React, { useState } from 'react';
import { downloadCertificate, printCertificate, generateShareableLink } from '../../lib/certificateGenerator';

/**
 * Certificate - Display and manage completion certificate
 */
const Certificate = ({ certificate }) => {
  const [copied, setCopied] = useState(false);

  const handleDownload = () => {
    downloadCertificate(certificate);
  };

  const handlePrint = () => {
    printCertificate(certificate);
  };

  const handleShare = () => {
    const shareUrl = generateShareableLink(certificate);
    navigator.clipboard.writeText(shareUrl);
    setCopied(true);
    setTimeout(() => setCopied(false), 3000);
  };

  return (
    <div className="space-y-6">
      {/* Certificate Preview */}
      <div className="bg-gradient-to-br from-blue-600 to-purple-600 rounded-lg shadow-2xl p-8">
        <div className="bg-white rounded-lg p-12 relative">
          {/* Decorative border */}
          <div className="absolute inset-4 border-4 border-blue-600 rounded pointer-events-none opacity-20" />

          {/* Content */}
          <div className="relative z-10 space-y-6">
            {/* Header */}
            <div className="text-center">
              <div className="text-5xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent mb-2">
                JadeVectorDB
              </div>
              <div className="text-2xl text-gray-700 mb-1">
                Tutorial Certification Program
              </div>
              <div className="text-sm text-gray-500 uppercase tracking-wide">
                Certificate of Completion
              </div>
            </div>

            {/* Recipient */}
            <div className="text-center my-8">
              <div className="text-sm text-gray-600 mb-2">This certifies that</div>
              <div className="text-4xl font-serif font-bold text-gray-800 border-b-2 border-gray-300 pb-2 inline-block">
                {certificate.userName}
              </div>
            </div>

            {/* Achievement */}
            <div className="text-center text-gray-700 leading-relaxed">
              <p>has successfully completed the JadeVectorDB Tutorial Program</p>
              <p>and demonstrated <strong className="text-blue-600">{certificate.proficiencyLevel}</strong> proficiency</p>
              <p>in vector database concepts and operations</p>
            </div>

            {/* Stats */}
            <div className="grid grid-cols-3 gap-6 my-8">
              <div className="text-center p-4 bg-blue-50 rounded-lg">
                <div className="text-3xl font-bold text-blue-600">{certificate.overallScore}%</div>
                <div className="text-xs text-gray-600 uppercase mt-1">Overall Score</div>
              </div>
              <div className="text-center p-4 bg-green-50 rounded-lg">
                <div className="text-3xl font-bold text-green-600">
                  {certificate.completedModules}/{certificate.totalModules}
                </div>
                <div className="text-xs text-gray-600 uppercase mt-1">Modules Completed</div>
              </div>
              <div className="text-center p-4 bg-purple-50 rounded-lg">
                <div className="text-2xl font-bold text-purple-600">
                  {certificate.certificationLevel.replace('JadeVectorDB ', '')}
                </div>
                <div className="text-xs text-gray-600 uppercase mt-1">Certification Level</div>
              </div>
            </div>

            {/* Skill Areas */}
            <div className="my-6">
              <h4 className="text-center text-sm font-semibold text-gray-700 mb-3 uppercase">
                Skill Areas Mastered
              </h4>
              <div className="grid grid-cols-2 gap-3">
                {certificate.skillAreas.map((skill, index) => (
                  <div key={index} className="flex items-center justify-between p-2 bg-gray-50 rounded">
                    <span className="text-sm text-gray-700">{skill.name}</span>
                    <span className={`text-sm font-bold ${
                      skill.status === 'strong' ? 'text-green-600' :
                      skill.status === 'moderate' ? 'text-yellow-600' : 'text-orange-600'
                    }`}>
                      {skill.score}%
                    </span>
                  </div>
                ))}
              </div>
            </div>

            {/* Footer */}
            <div className="text-center pt-6 border-t border-gray-300">
              <div className="text-sm text-gray-600 mb-4">
                <strong>Certificate ID:</strong> {certificate.certificateId}
              </div>
              <div className="text-sm text-gray-600 mb-2">
                <strong>Issue Date:</strong> {certificate.issueDateFormatted}
              </div>
              <div className="text-xs text-gray-500">
                Verify at: {certificate.verificationUrl}
              </div>
            </div>

            {/* Signature */}
            <div className="flex justify-center mt-8">
              <div className="text-center">
                <div className="border-t-2 border-gray-800 pt-2 px-12 font-serif font-bold text-gray-800">
                  JadeVectorDB Team
                </div>
                <div className="text-xs text-gray-600 mt-1">Tutorial Program Directors</div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Actions */}
      <div className="bg-white rounded-lg shadow p-6">
        <h3 className="text-xl font-bold text-gray-800 mb-4">Certificate Actions</h3>
        <div className="grid md:grid-cols-3 gap-4">
          <button
            onClick={handleDownload}
            className="p-4 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors flex items-center justify-center gap-2"
          >
            <span className="text-2xl">⬇️</span>
            <span className="font-semibold">Download HTML</span>
          </button>

          <button
            onClick={handlePrint}
            className="p-4 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors flex items-center justify-center gap-2"
          >
            <span className="text-2xl">🖨️</span>
            <span className="font-semibold">Print</span>
          </button>

          <button
            onClick={handleShare}
            className="p-4 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition-colors flex items-center justify-center gap-2"
          >
            <span className="text-2xl">🔗</span>
            <span className="font-semibold">{copied ? 'Copied!' : 'Copy Link'}</span>
          </button>
        </div>

        <div className="mt-4 p-4 bg-yellow-50 border border-yellow-200 rounded-lg">
          <p className="text-sm text-yellow-800">
            <strong>💡 Tip:</strong> Download your certificate to keep a permanent record, or share the verification link on your professional profiles!
          </p>
        </div>
      </div>

      {/* Certificate Details */}
      <div className="bg-white rounded-lg shadow p-6">
        <h3 className="text-xl font-bold text-gray-800 mb-4">Certificate Details</h3>
        <dl className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <dt className="text-sm font-semibold text-gray-600">Certificate ID</dt>
            <dd className="text-gray-800 font-mono">{certificate.certificateId}</dd>
          </div>
          <div>
            <dt className="text-sm font-semibold text-gray-600">Issue Date</dt>
            <dd className="text-gray-800">{certificate.issueDateFormatted}</dd>
          </div>
          <div>
            <dt className="text-sm font-semibold text-gray-600">Program Name</dt>
            <dd className="text-gray-800">{certificate.programName}</dd>
          </div>
          <div>
            <dt className="text-sm font-semibold text-gray-600">Issuing Organization</dt>
            <dd className="text-gray-800">{certificate.issuingOrganization}</dd>
          </div>
          <div>
            <dt className="text-sm font-semibold text-gray-600">Certification Level</dt>
            <dd className="text-gray-800 font-bold">{certificate.certificationLevel}</dd>
          </div>
          <div>
            <dt className="text-sm font-semibold text-gray-600">Validity Period</dt>
            <dd className="text-gray-800">{certificate.validityPeriod}</dd>
          </div>
        </dl>

        <div className="mt-6 p-4 bg-blue-50 rounded-lg">
          <p className="text-sm text-blue-800">
            <strong>🔒 Verification:</strong> This certificate can be verified at{' '}
            <a
              href={certificate.verificationUrl}
              target="_blank"
              rel="noopener noreferrer"
              className="underline font-semibold"
            >
              {certificate.verificationUrl}
            </a>
          </p>
        </div>
      </div>

      {/* Share on Social Media */}
      <div className="bg-white rounded-lg shadow p-6">
        <h3 className="text-xl font-bold text-gray-800 mb-4">Share Your Achievement</h3>
        <p className="text-gray-600 mb-4">
          Celebrate your achievement and share your JadeVectorDB certification with your professional network!
        </p>
        <div className="flex gap-3">
          <a
            href={`https://www.linkedin.com/sharing/share-offsite/?url=${encodeURIComponent(certificate.verificationUrl)}`}
            target="_blank"
            rel="noopener noreferrer"
            className="px-4 py-2 bg-[#0077b5] text-white rounded-lg hover:bg-[#006396] transition-colors"
          >
            Share on LinkedIn
          </a>
          <a
            href={`https://twitter.com/intent/tweet?text=${encodeURIComponent(`I just earned my ${certificate.certificationLevel} certification! 🎉`)}&url=${encodeURIComponent(certificate.verificationUrl)}`}
            target="_blank"
            rel="noopener noreferrer"
            className="px-4 py-2 bg-[#1DA1F2] text-white rounded-lg hover:bg-[#1a8cd8] transition-colors"
          >
            Share on Twitter
          </a>
        </div>
      </div>
    </div>
  );
};

export default Certificate;
