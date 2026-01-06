import React, { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { Eye, LogIn, User, Stethoscope, ArrowRight, Sparkles } from 'lucide-react';
import { useAuth } from '../context/AuthContext';

const Login = () => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [role, setRole] = useState<'patient' | 'doctor'>('patient');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const navigate = useNavigate();
  const { login } = useAuth();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError('');

    const success = await login(email, password, role);

    if (success) {
      navigate(`/${role}-dashboard`);
    } else {
      setError('Login failed. Please try again.');
    }

    setLoading(false);
  };

  return (
    <div className="min-h-screen bg-white flex items-center justify-center py-12 px-4 sm:px-6 lg:px-8 relative overflow-hidden">
      {/* Background decorative elements */}
      <div className="absolute inset-0 opacity-5">
        <div className="absolute top-20 left-20 w-96 h-96 bg-primary-500 rounded-full blur-3xl"></div>
        <div className="absolute bottom-20 right-20 w-96 h-96 bg-secondary-500 rounded-full blur-3xl"></div>
      </div>

      <div className="max-w-md w-full space-y-8 relative z-10 animate-fade-in-up">
        <div className="text-center">
          <Link to="/" className="inline-flex items-center space-x-3 mb-8 group transition-all duration-300 hover:scale-105">
            <div className="relative">
              <div className="absolute inset-0 bg-secondary-500 rounded-xl blur-md opacity-0 group-hover:opacity-50 transition-opacity duration-300"></div>
              <div className="relative bg-primary-500 rounded-xl p-3 shadow-lg transform transition-all duration-300 group-hover:rotate-6">
                <Eye className="h-8 w-8 text-white" />
              </div>
            </div>
            <div className="flex flex-col text-left">
              <span className="text-3xl font-bold text-primary-500 group-hover:text-secondary-500 transition-colors duration-300">
                RetinalAI
              </span>
              <span className="text-xs text-gray-500 font-medium -mt-1">AI Healthcare</span>
            </div>
          </Link>
          <h2 className="text-4xl font-bold text-gray-900 mb-3">Welcome Back</h2>
          <p className="text-lg text-gray-500">Sign in to access your personalized dashboard</p>
        </div>

        <div className="bg-white shadow-2xl rounded-3xl p-8 border-2 border-gray-100 animate-scale-in">
          {/* Role Selection */}
          <div className="mb-8">
            <label className="block text-sm font-semibold text-gray-900 mb-4">Login as:</label>
            <div className="grid grid-cols-2 gap-4">
              <button
                type="button"
                onClick={() => setRole('patient')}
                className={`p-4 rounded-xl border-2 transition-all duration-300 transform hover:scale-105 ${
                  role === 'patient'
                    ? 'border-primary-500 bg-primary-50 text-primary-700 shadow-md'
                    : 'border-gray-200 hover:border-primary-300 text-gray-900 hover:bg-gray-50'
                }`}
              >
                <User className={`h-6 w-6 mx-auto mb-2 ${role === 'patient' ? 'text-primary-500' : 'text-gray-400'}`} />
                <span className="text-sm font-semibold block">Patient</span>
              </button>
              <button
                type="button"
                onClick={() => setRole('doctor')}
                className={`p-4 rounded-xl border-2 transition-all duration-300 transform hover:scale-105 ${
                  role === 'doctor'
                    ? 'border-primary-500 bg-primary-50 text-primary-700 shadow-md'
                    : 'border-gray-200 hover:border-primary-300 text-gray-900 hover:bg-gray-50'
                }`}
              >
                <Stethoscope className={`h-6 w-6 mx-auto mb-2 ${role === 'doctor' ? 'text-primary-500' : 'text-gray-400'}`} />
                <span className="text-sm font-semibold block">Doctor</span>
              </button>
            </div>
          </div>

          <form onSubmit={handleSubmit} className="space-y-6">
            {error && (
              <div className="bg-red-50 border-2 border-red-200 text-red-700 px-4 py-3 rounded-xl text-sm font-medium animate-slide-down">
                {error}
              </div>
            )}

            <div>
              <label htmlFor="email" className="block text-sm font-semibold text-gray-900 mb-2">
                Email address
              </label>
              <input
                id="email"
                name="email"
                type="email"
                autoComplete="email"
                required
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                className="w-full px-4 py-3.5 border-2 border-gray-200 rounded-xl focus:ring-2 focus:ring-primary-500 focus:border-primary-500 transition-all duration-300 text-gray-900 placeholder-gray-400"
                placeholder="Enter your email"
              />
            </div>

            <div>
              <label htmlFor="password" className="block text-sm font-semibold text-gray-900 mb-2">
                Password
              </label>
              <input
                id="password"
                name="password"
                type="password"
                autoComplete="current-password"
                required
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className="w-full px-4 py-3.5 border-2 border-gray-200 rounded-xl focus:ring-2 focus:ring-primary-500 focus:border-primary-500 transition-all duration-300 text-gray-900 placeholder-gray-400"
                placeholder="Enter your password"
              />
            </div>

            <button
              type="submit"
              disabled={loading}
              className="w-full flex justify-center items-center py-4 px-4 border border-transparent rounded-xl shadow-lg text-base font-bold text-white bg-secondary-500 hover:bg-secondary-600 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-secondary-500 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-300 transform hover:scale-105 group"
            >
              {loading ? (
                <>
                  <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white mr-2"></div>
                  Signing in...
                </>
              ) : (
                <>
                  <LogIn className="h-5 w-5 mr-2" />
                  <span>Sign in</span>
                  <ArrowRight className="h-5 w-5 ml-2 group-hover:translate-x-1 transition-transform" />
                </>
              )}
            </button>

            <div className="text-center pt-4">
              <p className="text-sm text-gray-500">
                Don't have an account?{' '}
                <Link to="/register" className="font-bold text-secondary-500 hover:text-secondary-600 transition-colors duration-300 inline-flex items-center space-x-1 group">
                  <span>Register here</span>
                  <ArrowRight className="h-4 w-4 group-hover:translate-x-1 transition-transform" />
                </Link>
              </p>
            </div>
          </form>
        </div>
      </div>
    </div>
  );
};

export default Login;
