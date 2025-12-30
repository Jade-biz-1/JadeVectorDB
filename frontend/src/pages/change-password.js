import Head from 'next/head';
import { useState, useEffect } from 'react';
import { useRouter } from 'next/router';
import Link from 'next/link';
import { Button } from '../components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../components/ui/card';
import { Input } from '../components/ui/input';
import { Alert, AlertTitle, AlertDescription } from '../components/ui/alert';
import { authApi } from '../lib/api';

export default function ChangePassword() {
  const router = useRouter();
  const [oldPassword, setOldPassword] = useState('');
  const [newPassword, setNewPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  const [passwordStrength, setPasswordStrength] = useState({ score: 0, message: '' });
  const [mustChangePassword, setMustChangePassword] = useState(false);

  useEffect(() => {
    // Check if user is logged in
    const token = localStorage.getItem('jadevectordb_auth_token');
    if (!token) {
      router.push('/login');
      return;
    }

    // Check if user must change password
    const mustChange = localStorage.getItem('jadevectordb_must_change_password') === 'true';
    setMustChangePassword(mustChange);
  }, [router]);

  const validatePasswordStrength = (password) => {
    let score = 0;
    let message = '';

    if (password.length < 10) {
      return { score: 0, message: 'Password must be at least 10 characters' };
    }
    score += 1;

    if (/[A-Z]/.test(password)) score += 1;
    if (/[a-z]/.test(password)) score += 1;
    if (/[0-9]/.test(password)) score += 1;
    if (/[^A-Za-z0-9]/.test(password)) score += 1;

    if (score < 5) {
      message = 'Password must contain uppercase, lowercase, digit, and special character';
    } else if (password.length < 12) {
      message = 'Good - Consider using 12+ characters for better security';
    } else {
      message = 'Strong password';
    }

    return { score, message };
  };

  const handlePasswordChange = (e) => {
    const password = e.target.value;
    setNewPassword(password);

    if (password) {
      const strength = validatePasswordStrength(password);
      setPasswordStrength(strength);
    } else {
      setPasswordStrength({ score: 0, message: '' });
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setSuccess('');
    setLoading(true);

    try {
      // Validate input
      if (!oldPassword || !newPassword || !confirmPassword) {
        setError('Please fill in all fields');
        setLoading(false);
        return;
      }

      // Validate new password matches confirmation
      if (newPassword !== confirmPassword) {
        setError('New passwords do not match');
        setLoading(false);
        return;
      }

      // Validate new password is different from old password
      if (oldPassword === newPassword) {
        setError('New password must be different from old password');
        setLoading(false);
        return;
      }

      // Validate password strength
      const strength = validatePasswordStrength(newPassword);
      if (strength.score < 5) {
        setError(strength.message);
        setLoading(false);
        return;
      }

      // Get user ID from localStorage
      const userId = localStorage.getItem('jadevectordb_user_id');
      if (!userId) {
        setError('User ID not found. Please log in again.');
        setLoading(false);
        return;
      }

      // Call the password change API
      await authApi.changePassword(userId, oldPassword, newPassword);

      setSuccess('Password changed successfully! Redirecting to dashboard...');

      // Clear the must_change_password flag (already done in authApi.changePassword)
      // Redirect to dashboard after successful change
      setTimeout(() => {
        router.push('/dashboard');
      }, 1500);

    } catch (err) {
      console.error('Password change error:', err);
      setError(err.message || 'Failed to change password. Please check your old password.');
    } finally {
      setLoading(false);
    }
  };

  const handleLogout = () => {
    authApi.logout();
    router.push('/login');
  };

  return (
    <>
      <Head>
        <title>Change Password - JadeVectorDB</title>
        <meta name="description" content="Change your password" />
      </Head>

      <div className="min-h-screen flex items-center justify-center bg-gray-50 py-12 px-4 sm:px-6 lg:px-8">
        <div className="max-w-md w-full space-y-8">
          <div className="text-center">
            <h1 className="text-4xl font-bold text-gray-900 mb-2">JadeVectorDB</h1>
            <p className="text-gray-600">Change Your Password</p>
          </div>

          {mustChangePassword && (
            <Alert variant="warning">
              <AlertTitle>Password Change Required</AlertTitle>
              <AlertDescription>
                You must change your password before continuing to use the system.
              </AlertDescription>
            </Alert>
          )}

          <Card>
            <CardHeader>
              <CardTitle>Change Password</CardTitle>
              <CardDescription>
                Enter your current password and choose a new one
              </CardDescription>
            </CardHeader>
            <CardContent>
              <form onSubmit={handleSubmit} className="space-y-4">
                <div>
                  <label htmlFor="oldPassword" className="block text-sm font-medium text-gray-700 mb-1">
                    Current Password
                  </label>
                  <Input
                    id="oldPassword"
                    type="password"
                    placeholder="Enter your current password"
                    value={oldPassword}
                    onChange={(e) => setOldPassword(e.target.value)}
                    disabled={loading}
                    required
                  />
                </div>

                <div>
                  <label htmlFor="newPassword" className="block text-sm font-medium text-gray-700 mb-1">
                    New Password
                  </label>
                  <Input
                    id="newPassword"
                    type="password"
                    placeholder="Enter your new password"
                    value={newPassword}
                    onChange={handlePasswordChange}
                    disabled={loading}
                    required
                  />
                  {passwordStrength.message && (
                    <p className={`text-xs mt-1 ${
                      passwordStrength.score === 5
                        ? 'text-green-600'
                        : passwordStrength.score >= 3
                          ? 'text-yellow-600'
                          : 'text-red-600'
                    }`}>
                      {passwordStrength.message}
                    </p>
                  )}
                  <p className="text-xs text-gray-500 mt-1">
                    Must be at least 10 characters with uppercase, lowercase, digit, and special character
                  </p>
                </div>

                <div>
                  <label htmlFor="confirmPassword" className="block text-sm font-medium text-gray-700 mb-1">
                    Confirm New Password
                  </label>
                  <Input
                    id="confirmPassword"
                    type="password"
                    placeholder="Re-enter your new password"
                    value={confirmPassword}
                    onChange={(e) => setConfirmPassword(e.target.value)}
                    disabled={loading}
                    required
                  />
                </div>

                {error && (
                  <Alert variant="destructive">
                    <AlertTitle>Error</AlertTitle>
                    <AlertDescription>{error}</AlertDescription>
                  </Alert>
                )}

                {success && (
                  <Alert variant="success">
                    <AlertTitle>Success</AlertTitle>
                    <AlertDescription>{success}</AlertDescription>
                  </Alert>
                )}

                <Button
                  type="submit"
                  className="w-full"
                  disabled={loading}
                >
                  {loading ? 'Changing Password...' : 'Change Password'}
                </Button>
              </form>

              <div className="mt-6 text-center space-y-2">
                {!mustChangePassword && (
                  <div>
                    <Button
                      variant="outline"
                      onClick={handleLogout}
                      className="w-full"
                    >
                      Cancel and Logout
                    </Button>
                  </div>
                )}
                {mustChangePassword && (
                  <p className="text-sm text-gray-600">
                    You must change your password to continue using the system
                  </p>
                )}
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </>
  );
}
