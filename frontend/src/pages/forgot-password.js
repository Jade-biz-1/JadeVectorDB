import Head from 'next/head';
import { useState } from 'react';
import Link from 'next/link';
import { Button } from '../components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../components/ui/card';
import { Input } from '../components/ui/input';
import { Alert, AlertTitle, AlertDescription } from '../components/ui/alert';
import { authApi } from '../lib/api';

export default function ForgotPassword() {
  const [username, setUsername] = useState('');
  const [email, setEmail] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');

  const handleForgotPassword = async (e) => {
    e.preventDefault();
    setError('');
    setSuccess('');

    if (!username && !email) {
      setError('Please enter either your username or email');
      return;
    }

    setLoading(true);

    try {
      // Call the forgot password API
      const response = await authApi.forgotPassword(username, email);

      setSuccess('Password reset instructions have been sent. Please check the response for your reset token.');

      // In a real application, the reset token would be sent via email
      // For now, we'll display it in the success message
      if (response.reset_token) {
        setSuccess(
          `Password reset initiated. Reset Token: ${response.reset_token}. ` +
          `User ID: ${response.user_id}. Please save these and use them on the reset password page.`
        );
      }

      // Clear form
      setUsername('');
      setEmail('');

    } catch (err) {
      console.error('Forgot password error:', err);
      setError(err.message || 'Failed to process password reset request. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      <Head>
        <title>Forgot Password - JadeVectorDB</title>
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
              <CardTitle>Forgot Password</CardTitle>
              <CardDescription>
                Enter your username or email to receive password reset instructions
              </CardDescription>
            </CardHeader>
            <CardContent>
              <form onSubmit={handleForgotPassword} className="space-y-4">
                <div>
                  <label htmlFor="username" className="block text-sm font-medium text-gray-700 mb-1">
                    Username
                  </label>
                  <Input
                    id="username"
                    type="text"
                    placeholder="Enter your username"
                    value={username}
                    onChange={(e) => setUsername(e.target.value)}
                    disabled={loading}
                  />
                </div>

                <div className="text-center text-sm text-gray-500">
                  OR
                </div>

                <div>
                  <label htmlFor="email" className="block text-sm font-medium text-gray-700 mb-1">
                    Email
                  </label>
                  <Input
                    id="email"
                    type="email"
                    placeholder="Enter your email"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    disabled={loading}
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
                    <AlertDescription className="whitespace-pre-wrap">{success}</AlertDescription>
                  </Alert>
                )}

                <Button
                  type="submit"
                  className="w-full"
                  disabled={loading}
                >
                  {loading ? 'Processing...' : 'Send Reset Instructions'}
                </Button>
              </form>

              <div className="mt-6 text-center space-y-2">
                <div>
                  <span className="text-sm text-gray-600">Remember your password? </span>
                  <Link href="/login" className="text-sm text-blue-600 hover:text-blue-500">
                    Sign in
                  </Link>
                </div>
                <div>
                  <Link href="/reset-password" className="text-sm text-blue-600 hover:text-blue-500">
                    Already have a reset token? →
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
