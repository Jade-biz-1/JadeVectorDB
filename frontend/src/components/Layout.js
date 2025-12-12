import { useRouter } from 'next/router';
import { useState, useEffect } from 'react';
import { authApi } from '../lib/api';
import Head from 'next/head';

export default function Layout({ children, title = 'JadeVectorDB' }) {
  const router = useRouter();
  const [menuOpen, setMenuOpen] = useState(false);
  const [username, setUsername] = useState('');

  useEffect(() => {
    // Only access localStorage on the client side
    if (typeof window !== 'undefined') {
      setUsername(localStorage.getItem('jadevectordb_username') || '');
    }
  }, []);

  const handleLogout = async () => {
    try {
      await authApi.logout();
    } catch (error) {
      console.error('Logout error:', error);
    } finally {
      router.push('/');
    }
  };

  const navItems = [
    { name: 'Dashboard', href: '/dashboard' },
    { name: 'Databases', href: '/databases' },
    { name: 'Search', href: '/search' },
    { name: 'Users', href: '/users' },
    { name: 'API Keys', href: '/api-keys' },
    { name: 'Monitoring', href: '/monitoring' },
  ];

  return (
    <>
      <Head>
        <title>{title}</title>
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <style>{`
          * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
          }

          body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen', 'Ubuntu', 'Cantarell', sans-serif;
            background: #f9fafb;
            color: #111827;
            line-height: 1.6;
          }

          .nav-container {
            background: white;
            border-bottom: 1px solid #e5e7eb;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
          }

          .nav-inner {
            max-width: 1280px;
            margin: 0 auto;
            padding: 0 1rem;
          }

          .nav-content {
            display: flex;
            justify-content: space-between;
            align-items: center;
            height: 64px;
          }

          .nav-brand {
            font-size: 1.5rem;
            font-weight: 700;
            color: #667eea;
            text-decoration: none;
          }

          .nav-links {
            display: none;
            gap: 2rem;
          }

          @media (min-width: 768px) {
            .nav-links {
              display: flex;
            }
          }

          .nav-link {
            color: #6b7280;
            text-decoration: none;
            font-weight: 500;
            padding: 0.5rem 0;
            transition: color 0.2s;
          }

          .nav-link:hover {
            color: #667eea;
          }

          .nav-link.active {
            color: #667eea;
            border-bottom: 2px solid #667eea;
          }

          .nav-right {
            display: flex;
            align-items: center;
            gap: 1rem;
          }

          .user-info {
            display: none;
            color: #6b7280;
            font-size: 0.875rem;
          }

          @media (min-width: 768px) {
            .user-info {
              display: block;
            }
          }

          .btn-logout {
            padding: 0.5rem 1rem;
            background: #ef4444;
            color: white;
            border: none;
            border-radius: 6px;
            font-weight: 500;
            cursor: pointer;
            transition: background 0.2s;
          }

          .btn-logout:hover {
            background: #dc2626;
          }

          .hamburger {
            display: flex;
            flex-direction: column;
            gap: 4px;
            background: none;
            border: none;
            cursor: pointer;
            padding: 0.5rem;
          }

          @media (min-width: 768px) {
            .hamburger {
              display: none;
            }
          }

          .hamburger span {
            display: block;
            width: 24px;
            height: 2px;
            background: #6b7280;
            transition: all 0.3s;
          }

          .mobile-menu {
            display: none;
            background: white;
            border-top: 1px solid #e5e7eb;
            padding: 1rem;
          }

          .mobile-menu.open {
            display: block;
          }

          .mobile-link {
            display: block;
            padding: 0.75rem 1rem;
            color: #6b7280;
            text-decoration: none;
            border-radius: 6px;
            transition: background 0.2s;
          }

          .mobile-link:hover {
            background: #f3f4f6;
            color: #667eea;
          }

          .main-content {
            max-width: 1280px;
            margin: 0 auto;
            padding: 2rem 1rem;
          }
        `}</style>
      </Head>

      <nav className="nav-container">
        <div className="nav-inner">
          <div className="nav-content">
            <a href="/dashboard" className="nav-brand">JadeVectorDB</a>

            <div className="nav-links">
              {navItems.map((item) => (
                <a
                  key={item.href}
                  href={item.href}
                  className={`nav-link ${router.pathname === item.href ? 'active' : ''}`}
                >
                  {item.name}
                </a>
              ))}
            </div>

            <div className="nav-right">
              <span className="user-info">
                {username}
              </span>
              <button onClick={handleLogout} className="btn-logout">
                Logout
              </button>
              <button className="hamburger" onClick={() => setMenuOpen(!menuOpen)}>
                <span></span>
                <span></span>
                <span></span>
              </button>
            </div>
          </div>
        </div>

        <div className={`mobile-menu ${menuOpen ? 'open' : ''}`}>
          {navItems.map((item) => (
            <a key={item.href} href={item.href} className="mobile-link">
              {item.name}
            </a>
          ))}
        </div>
      </nav>

      <main className="main-content">
        {children}
      </main>
    </>
  );
}
