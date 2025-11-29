import Head from 'next/head';
import { useState } from 'react';
import { useRouter } from 'next/router';
import Link from 'next/link';
import { Button } from '../components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../components/ui/card';
import { Input } from '../components/ui/input';
import { Alert, AlertTitle, AlertDescription } from '../components/ui/alert';
import { authApi } from '../lib/api';

export default function ResetPassword() {
  const router = useRouter();
  const [formData, setFormData] = useState({
    user_id: '',
    reset_token: '',
    new_password: '',
    confirm_password: '',
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');

  const handleChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value,
    });
  };

  const validateForm = () => {
    if (!formData.user_id || !formData.reset_token || !formData.new_password || !formData.confirm_password) {
      setError('Please fill in all fields');
      return false;
    }

    if (formData.new_password.length < 8) {
      setError('Password must be at least 8 characters long');
      return false;
    }

    if (formData.new_password !== formData.confirm_password) {
      setError('Passwords do not match');
      return false;
    }

    return true;
  };

  const handleResetPassword = async (e) => {
    e.preventDefault();
    setError('');
    setSuccess('');

    if (!validateForm()) {
      return;
    }

    setLoading(true);

    try {
      // Call the reset password API
      await authApi.resetPassword(
        formData.user_id,
        formData.reset_token,
        formData.new_password
      );

      setSuccess('Password reset successfully! Redirecting to login...');

      // Redirect to login page after successful password reset
      setTimeout(() => {
        router.push('/login');
      }, 2000);

    } catch (err) {
      console.error('Reset password error:', err);
      setError(err.message || 'Failed to reset password. Please check your reset token and try again.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      <Head>
        <title>Reset Password - JadeVectorDB</title>
        <meta name="description" content="Reset your JadeVectorDB password" />
      </Head>

      <div className="min-h-screen flex items-center justify-center bg-gray-50 py-12 px-4 sm:px-6 lg:px-8">
        <div className="max-w-md w-full space-y-8">
          <div className="text-center">
            <h1 className="text-4xl font-bold text-gray-900 mb-2">JadeVectorDB</h1>
            <p className="text-gray-600">Reset your password</p>
          </div>

          <Card>
            <CardHeader>
              <CardTitle>Reset Password</CardTitle>
              <CardDescription>
                Enter your reset token and new password
              </CardDescription>
            </CardHeader>
            <CardContent>
              <form onSubmit={handleResetPassword} className="space-y-4">
                <div>
                  <label htmlFor="user_id" className="block text-sm font-medium text-gray-700 mb-1">
                    User ID <span className="text-red-500">*</span>
                  </label>
                  <Input
                    id="user_id"
                    name="user_id"
                    type="text"
                    placeholder="Enter your user ID"
                    value={formData.user_id}
                    onChange={handleChange}
                    disabled={loading}
                    required
                  />
                  <p className="text-xs text-gray-500 mt-1">
                    User ID from the password reset email/response
                  </p>
                </div>

                <div>
                  <label htmlFor="reset_token" className="block text-sm font-medium text-gray-700 mb-1">
                    Reset Token <span className="text-red-500">*</span>
                  </label>
                  <Input
                    id="reset_token"
                    name="reset_token"
                    type="text"
                    placeholder="Enter your reset token"
                    value={formData.reset_token}
                    onChange={handleChange}
                    disabled={loading}
                    required
                  />
                  <p className="text-xs text-gray-500 mt-1">
                    Reset token from the password reset email/response
                  </p>
                </div>

                <div>
                  <label htmlFor="new_password" className="block text-sm font-medium text-gray-700 mb-1">
                    New Password <span className="text-red-500">*</span>
                  </label>
                  <Input
                    id="new_password"
                    name="new_password"
                    type="password"
                    placeholder="Enter new password"
                    value={formData.new_password}
                    onChange={handleChange}
                    disabled={loading}
                    required
                  />
                  <p className="text-xs text-gray-500 mt-1">At least 8 characters</p>
                </div>

                <div>
                  <label htmlFor="confirm_password" className="block text-sm font-medium text-gray-700 mb-1">
                    Confirm Password <span className="text-red-500">*</span>
                  </label>
                  <Input
                    id="confirm_password"
                    name="confirm_password"
                    type="password"
                    placeholder="Confirm new password"
                    value={formData.confirm_password}
                    onChange={handleChange}
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
                  {loading ? 'Resetting password...' : 'Reset Password'}
                </Button>
              </form>

              <div className="mt-6 text-center space-y-2">
                <div>
                  <Link href="/forgot-password" className="text-sm text-blue-600 hover:text-blue-500">
                    ← Request new reset token
                  </Link>
                </div>
                <div>
                  <span className="text-sm text-gray-600">Remember your password? </span>
                  <Link href="/login" className="text-sm text-blue-600 hover:text-blue-500">
                    Sign in
                  </Link>
                </div>
                <div className="pt-4 border-t mt-4">
                  <Link href="/" className="text-sm text-gray-600 hover:text-gray-500">
                    ← Back to home
                  </Link>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </>
  );
}
