import { useState } from 'react';
import { adminAPI } from '../services/api';

function DocumentUpload({ onSuccess }) {
  const [file, setFile] = useState(null);
  const [deviceType, setDeviceType] = useState('hydraulic_pump');
  const [uploading, setUploading] = useState(false);
  const [message, setMessage] = useState(null);

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
    setMessage(null);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!file) {
      setMessage({ type: 'error', text: 'Please select a file' });
      return;
    }

    setUploading(true);
    setMessage(null);

    try {
      const result = await adminAPI.uploadDocument(file, deviceType);
      setMessage({
        type: 'success',
        text: `Document "${file.name}" uploaded successfully. Processing...`,
      });
      setFile(null);
      // Reset file input
      e.target.reset();
      onSuccess();
    } catch (err) {
      setMessage({
        type: 'error',
        text: err.response?.data?.detail || 'Failed to upload document',
      });
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="document-upload">
      <h3>Upload Document</h3>

      <form onSubmit={handleSubmit} className="upload-form">
        <div className="form-group">
          <label htmlFor="file">Select PDF or DOCX file</label>
          <input
            type="file"
            id="file"
            accept=".pdf,.docx"
            onChange={handleFileChange}
            disabled={uploading}
            required
          />
          {file && (
            <span className="file-info">
              {file.name} ({(file.size / 1024 / 1024).toFixed(2)} MB)
            </span>
          )}
        </div>

        <div className="form-group">
          <label htmlFor="device-type">Device Type</label>
          <select
            id="device-type"
            value={deviceType}
            onChange={(e) => setDeviceType(e.target.value)}
            disabled={uploading}
          >
            <option value="hydraulic_pump">Hydraulic Pump</option>
            <option value="air_compressor">Air Compressor</option>
            <option value="conveyor">Conveyor System</option>
            <option value="other">Other</option>
          </select>
        </div>

        <button type="submit" className="upload-button" disabled={uploading || !file}>
          {uploading ? 'Uploading...' : 'Upload Document'}
        </button>
      </form>

      {message && (
        <div className={`upload-message ${message.type}`}>
          {message.text}
        </div>
      )}
    </div>
  );
}

export default DocumentUpload;
