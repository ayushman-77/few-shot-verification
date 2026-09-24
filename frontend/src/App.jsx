import React, { useState, useRef } from 'react';
import axios from 'axios';
import { UploadCloud, CheckCircle, XCircle, Loader2 } from 'lucide-react';
import './index.css';

function App() {
  const [image1, setImage1] = useState(null);
  const [image2, setImage2] = useState(null);
  const [preview1, setPreview1] = useState('');
  const [preview2, setPreview2] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');

  const fileInput1 = useRef(null);
  const fileInput2 = useRef(null);

  const handleImageChange = (e, setImage, setPreview) => {
    const file = e.target.files[0];
    if (file) {
      setImage(file);
      const reader = new FileReader();
      reader.onloadend = () => {
        setPreview(reader.result);
      };
      reader.readAsDataURL(file);
      setResult(null);
      setError('');
    }
  };

  const handleVerify = async () => {
    if (!image1 || !image2) return;
    
    setLoading(true);
    setError('');
    
    const formData = new FormData();
    formData.append('image1', image1);
    formData.append('image2', image2);

    try {
      const response = await axios.post('http://127.0.0.1:8000/verify', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      setResult(response.data);
    } catch (err) {
      setError(err.message || 'Error verifying images. Make sure backend is running.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container">
      <header className="header">
        <h1>Few Shot Verification</h1>
        <p>High-Resolution Structural Image Verification</p>
      </header>

      <main>
        <div className="upload-section">
          <div className="upload-card">
            <h2>Reference Image</h2>
            <div className="drop-zone" onClick={() => fileInput1.current.click()}>
              {preview1 ? (
                <img src={preview1} alt="Preview 1" className="preview-image" />
              ) : (
                <>
                  <UploadCloud />
                  <p>Click to upload image</p>
                </>
              )}
            </div>
            <input
              type="file"
              ref={fileInput1}
              onChange={(e) => handleImageChange(e, setImage1, setPreview1)}
              className="file-input"
              accept="image/*"
            />
          </div>

          <div className="upload-card">
            <h2>Query Image</h2>
            <div className="drop-zone" onClick={() => fileInput2.current.click()}>
              {preview2 ? (
                <img src={preview2} alt="Preview 2" className="preview-image" />
              ) : (
                <>
                  <UploadCloud />
                  <p>Click to upload image</p>
                </>
              )}
            </div>
            <input
              type="file"
              ref={fileInput2}
              onChange={(e) => handleImageChange(e, setImage2, setPreview2)}
              className="file-input"
              accept="image/*"
            />
          </div>
        </div>

        <div className="action-section">
          <button 
            className="verify-btn" 
            onClick={handleVerify} 
            disabled={!image1 || !image2 || loading}
          >
            {loading ? (
              <><Loader2 className="spinner" /> Analyzing Geometry...</>
            ) : (
              'Verify Match'
            )}
          </button>
        </div>

        {error && (
          <div className="result-card mismatch" style={{ marginTop: '2rem' }}>
            {error}
          </div>
        )}

        {result && (
          <div className="result-card">
            <div className={`result-status ${result.match ? 'match' : 'mismatch'}`}>
              {result.match ? <CheckCircle size={40} /> : <XCircle size={40} />}
              {result.match ? 'Verified Match' : 'Entity Mismatch'}
            </div>
            
            <div className="metrics">
              <div className="metric">
                <span className="metric-label">Similarity</span>
                <span className="metric-value">{(result.similarity_score * 100).toFixed(1)}%</span>
              </div>
              <div className="metric">
                <span className="metric-label">Euclidean Dist</span>
                <span className="metric-value">{result.euclidean_distance.toFixed(4)}</span>
              </div>
            </div>
            <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem', marginTop: '1rem' }}>

            </p>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
