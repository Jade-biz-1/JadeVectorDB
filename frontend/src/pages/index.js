import Head from 'next/head';
import { useState } from 'react';
import { useRouter } from 'next/router';
import { authApi } from '../lib/api';

export default function Home() {
  const router = useRouter();
  const [isLogin, setIsLogin] = useState(true);
  const [formData, setFormData] = useState({
    username: '',
    password: '',
    email: ''
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setLoading(true);

    try {
      if (isLogin) {
        const response = await authApi.login(formData.username, formData.password);
        console.log('Login successful:', response);
        router.push('/databases');
      } else {
        const response = await authApi.register(
          formData.username,
          formData.password,
          formData.email
        );
        console.log('Registration successful:', response);
        await authApi.login(formData.username, formData.password);
        router.push('/databases');
      }
    } catch (err) {
      console.error('Auth error:', err);
      setError(err.message || 'Authentication failed');
    } finally {
      setLoading(false);
    }
  };

  const handleInputChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value
    });
  };

  return (
    <>
      <Head>
        <title>JadeVectorDB - Vector Database Platform</title>
        <meta name="description" content="JadeVectorDB - High-performance distributed vector database" />
        <style>{`
          * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
          }

          body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen', 'Ubuntu', 'Cantarell', sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
          }

          .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
          }

          .header {
            background: white;
            padding: 30px 40px;
            margin-bottom: 60px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
          }

          .header h1 {
            font-size: 36px;
            color: #667eea;
            margin-bottom: 5px;
          }

          .header p {
            color: #666;
            font-size: 14px;
          }

          .main-content {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 40px;
            align-items: start;
          }

          @media (max-width: 968px) {
            .main-content {
              grid-template-columns: 1fr;
            }
          }

          .info-section h2 {
            font-size: 42px;
            color: white;
            margin-bottom: 20px;
            line-height: 1.2;
          }

          .info-section p {
            font-size: 18px;
            color: rgba(255,255,255,0.9);
            margin-bottom: 30px;
            line-height: 1.6;
          }

          .features {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
          }

          .feature-card {
            background: rgba(255,255,255,0.15);
            backdrop-filter: blur(10px);
            padding: 20px;
            border-radius: 10px;
            border: 1px solid rgba(255,255,255,0.2);
          }

          .feature-card h3 {
            color: white;
            font-size: 16px;
            margin-bottom: 8px;
          }

          .feature-card p {
            color: rgba(255,255,255,0.8);
            font-size: 13px;
            margin: 0;
          }

          .auth-card {
            background: white;
            padding: 40px;
            border-radius: 15px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
          }

          .auth-card h3 {
            font-size: 28px;
            color: #333;
            margin-bottom: 10px;
          }

          .auth-card .subtitle {
            color: #666;
            margin-bottom: 30px;
            font-size: 15px;
          }

          .form-group {
            margin-bottom: 20px;
          }

          .form-group label {
            display: block;
            margin-bottom: 8px;
            color: #333;
            font-weight: 500;
            font-size: 14px;
          }

          .form-group input {
            width: 100%;
            padding: 12px 15px;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            font-size: 15px;
            transition: border-color 0.3s;
          }

          .form-group input:focus {
            outline: none;
            border-color: #667eea;
          }

          .form-group input:disabled {
            background: #f5f5f5;
            cursor: not-allowed;
          }

          .hint {
            font-size: 12px;
            color: #888;
            margin-top: 5px;
          }

          .error-message {
            background: #fee;
            border: 1px solid #fcc;
            color: #c00;
            padding: 12px;
            border-radius: 8px;
            margin-bottom: 20px;
            font-size: 14px;
          }

          .submit-btn {
            width: 100%;
            padding: 14px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
          }

          .submit-btn:hover:not(:disabled) {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
          }

          .submit-btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
          }

          .toggle-auth {
            text-align: center;
            margin-top: 25px;
          }

          .toggle-auth button {
            background: none;
            border: none;
            color: #667eea;
            font-size: 14px;
            cursor: pointer;
            text-decoration: underline;
          }

          .toggle-auth button:hover {
            color: #764ba2;
          }

          .footer {
            text-align: center;
            color: rgba(255,255,255,0.8);
            margin-top: 60px;
            font-size: 14px;
          }
        `}</style>
      </Head>

      <div className="container">
        <div className="header">
          <h1>JadeVectorDB</h1>
          <p>High-Performance Distributed Vector Database Platform</p>
        </div>

        <div className="main-content">
          <div className="info-section">
            <h2>Enterprise Vector Database Solution</h2>
            <p>
              Store, retrieve, and search billions of vector embeddings with sub-50ms latency.
              Built for AI applications, semantic search, and similarity matching at scale.
            </p>

            <div className="features">
              <div className="feature-card">
                <h3>⚡ Lightning Fast</h3>
                <p>&lt;50ms queries on 1M+ vectors</p>
              </div>
              <div className="feature-card">
                <h3>🔒 Secure</h3>
                <p>Enterprise-grade authentication</p>
              </div>
              <div className="feature-card">
                <h3>📈 Scalable</h3>
                <p>Distributed cluster architecture</p>
              </div>
              <div className="feature-card">
                <h3>🛠️ Developer Ready</h3>
                <p>RESTful API with full docs</p>
              </div>
            </div>
          </div>

          <div className="auth-card">
            <h3>{isLogin ? 'Welcome Back' : 'Get Started'}</h3>
            <p className="subtitle">
              {isLogin ? 'Sign in to your account' : 'Create your account'}
            </p>

            <form onSubmit={handleSubmit}>
              <div className="form-group">
                <label htmlFor="username">Username</label>
                <input
                  id="username"
                  name="username"
                  type="text"
                  required
                  value={formData.username}
                  onChange={handleInputChange}
                  placeholder="Enter your username"
                  disabled={loading}
                />
              </div>

              {!isLogin && (
                <div className="form-group">
                  <label htmlFor="email">Email Address</label>
                  <input
                    id="email"
                    name="email"
                    type="email"
                    required={!isLogin}
                    value={formData.email}
                    onChange={handleInputChange}
                    placeholder="your@email.com"
                    disabled={loading}
                  />
                </div>
              )}

              <div className="form-group">
                <label htmlFor="password">Password</label>
                <input
                  id="password"
                  name="password"
                  type="password"
                  required
                  value={formData.password}
                  onChange={handleInputChange}
                  placeholder={isLogin ? "Enter your password" : "Create a strong password"}
                  disabled={loading}
                />
                {!isLogin && (
                  <div className="hint">
                    Must be 8+ characters with uppercase, lowercase, number & symbol
                  </div>
                )}
              </div>

              {error && (
                <div className="error-message">
                  {error}
                </div>
              )}

              <button type="submit" className="submit-btn" disabled={loading}>
                {loading ? 'Processing...' : (isLogin ? 'Sign In' : 'Create Account')}
              </button>
            </form>

            <div className="toggle-auth">
              <button
                type="button"
                onClick={() => {
                  setIsLogin(!isLogin);
                  setError('');
                  setFormData({ username: '', password: '', email: '' });
                }}
              >
                {isLogin ? "Don't have an account? Sign up" : 'Already have an account? Sign in'}
              </button>
            </div>
          </div>
        </div>

        <div className="footer">
          © 2025 JadeVectorDB • Built with C++20 & Next.js
        </div>
      </div>
    </>
  );
}
