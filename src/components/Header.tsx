import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { Eye, LogOut, User, Stethoscope, Bell } from 'lucide-react';
import { useAuth } from '../context/AuthContext';

const Header = () => {
  const { user, logout } = useAuth();
  const location = useLocation();

  const handleLogout = () => {
    logout();
  };

  const getNavItems = () => {
    if (user?.role === 'patient') {
      return [
        { path: '/patient-dashboard', label: 'Dashboard' },
        { path: '/new-analysis', label: 'New Analysis' },
        { path: '/results', label: 'Results' },
        { path: '/appointments', label: 'Appointments' }
      ];
    } else {
      return [
        { path: '/doctor-dashboard', label: 'Dashboard' },
        { path: '/doctor-dashboard', label: 'Reports' },
        { path: '/doctor-dashboard', label: 'Patients' }
      ];
    }
  };

  return (
    <header className="bg-primary-500 shadow-sm border-b border-primary-600">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          {/* Logo */}
          <Link to="/" className="flex items-center space-x-2">
            <Eye className="h-8 w-8 text-white" />
            <span className="text-xl font-bold text-white">RetinalAI</span>
          </Link>

          {/* Navigation */}
          <nav className="hidden md:flex items-center space-x-8">
            {getNavItems().map((item) => (
              <Link
                key={item.path}
                to={item.path}
                className={`px-3 py-2 rounded-md text-sm font-medium transition-colors ${
                  location.pathname === item.path
                    ? 'text-white bg-primary-600'
                    : 'text-white hover:text-gold-500 hover:bg-primary-600'
                }`}
              >
                {item.label}
              </Link>
            ))}
          </nav>

          {/* User Menu */}
          <div className="flex items-center space-x-4">
            <button className="p-2 text-white hover:text-gold-500 hover:bg-primary-600 rounded-lg transition-colors">
              <Bell className="h-5 w-5" />
            </button>
            
            <div className="flex items-center space-x-3 px-3 py-2 bg-primary-600 rounded-lg">
              <div className="flex items-center space-x-2">
                {user?.role === 'doctor' ? (
                  <Stethoscope className="h-4 w-4 text-white" />
                ) : (
                  <User className="h-4 w-4 text-white" />
                )}
                <div>
                  <p className="text-sm font-medium text-white">{user?.username}</p>
                  <p className="text-xs text-white/80 capitalize">{user?.role}</p>
                </div>
              </div>
            </div>

            <button
              onClick={handleLogout}
              className="p-2 text-white hover:text-gold-500 hover:bg-primary-600 rounded-lg transition-colors"
              title="Logout"
            >
              <LogOut className="h-5 w-5" />
            </button>
          </div>
        </div>
      </div>
    </header>
  );
};

export default Header;