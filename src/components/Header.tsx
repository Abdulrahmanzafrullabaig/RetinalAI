import React, { useState } from 'react';
import { Link, useLocation, useNavigate } from 'react-router-dom';
import { Eye, LogOut, User, Stethoscope, Bell, Menu, X, Check } from 'lucide-react';
import { useAuth } from '../context/AuthContext';
import { API_URL } from '../config';

interface Notification {
  id: number;
  message: string;
  type: string;
  is_read: boolean;
  created_at: string;
}

const Header = () => {
  const { user, logout } = useAuth();
  const location = useLocation();
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [notifications, setNotifications] = useState<Notification[]>([]);
  const [showNotifications, setShowNotifications] = useState(false);

  // Fetch notifications
  const fetchNotifications = async () => {
    if (!user) return;
    try {
      const response = await fetch(`${API_URL}/api/notifications`, {
        headers: {
          'Authorization': `Bearer ${user.id}` // In a real app we'd use a token, but here we rely on cookie session
        },
        credentials: 'include'
      });
      if (response.ok) {
        const data = await response.json();
        setNotifications(data);
      }
    } catch (error) {
      console.error('Error fetching notifications:', error);
    }
  };

  React.useEffect(() => {
    fetchNotifications();
    // Poll every 30 seconds
    const interval = setInterval(fetchNotifications, 30000);
    return () => clearInterval(interval);
  }, [user]);

  const handleLogout = () => {
    logout();
    setMobileMenuOpen(false);
  };

  const navigate = useNavigate();

  const handleNotificationClick = (notification: Notification) => {
    setShowNotifications(false);
    if (notification.type === 'appointment_request') {
      navigate('/doctor-dashboard');
    } else if (notification.type === 'appointment_update') {
      navigate('/appointments');
    }
  };

  const toggleNotifications = async () => {
    if (!showNotifications) {
      // Opening notifications
      setShowNotifications(true);
      // Mark as read
      if (unreadCount > 0) {
        try {
          await fetch(`${API_URL}/api/notifications/mark-read`, {
            method: 'POST',
            credentials: 'include'
          });
          // Update local state to show all read
          setNotifications(prev => prev.map(n => ({ ...n, is_read: true })));
        } catch (error) {
          console.error('Error marking notifications as read:', error);
        }
      }
    } else {
      setShowNotifications(false);
    }
  };

  const unreadCount = notifications.filter(n => !n.is_read).length;

  const getNavItems = () => {
    if (user?.role === 'patient') {
      return [
        { path: '/patient-dashboard', label: 'Dashboard' },
        { path: '/new-analysis', label: 'New Analysis' },
        { path: '/results', label: 'Results' },
        { path: '/shared-reports', label: 'Shared Reports' },
        { path: '/appointments', label: 'Appointments' }
      ];
    } else {
      return [];
    }
  };

  return (
    <header className="bg-primary-500 shadow-lg border-b-2 border-primary-600 sticky top-0 z-50 backdrop-blur-sm">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-20">
          {/* Logo */}
          <Link
            to="/"
            className="flex items-center space-x-3 group transition-all duration-300 hover:scale-105"
            onClick={() => setMobileMenuOpen(false)}
          >
            <div className="relative">
              <div className="absolute inset-0 bg-secondary-500 rounded-xl blur-md opacity-0 group-hover:opacity-50 transition-opacity duration-300"></div>
              <div className="relative bg-white rounded-xl p-2.5 shadow-lg transform transition-all duration-300 group-hover:rotate-6">
                <Eye className="h-7 w-7 text-primary-500" />
              </div>
            </div>
            <div className="flex flex-col">
              <span className="text-2xl font-bold text-white tracking-tight group-hover:text-secondary-500 transition-colors duration-300">
                RetinalAI
              </span>
              <span className="text-xs text-white/80 font-medium -mt-1">AI Healthcare</span>
            </div>
          </Link>

          {/* Desktop Navigation */}
          <nav className="hidden md:flex items-center space-x-2">
            {getNavItems().map((item, index) => (
              <Link
                key={item.path}
                to={item.path}
                className={`relative px-4 py-2.5 rounded-lg text-sm font-semibold transition-all duration-300 transform hover:scale-105 ${location.pathname === item.path
                  ? 'text-white bg-primary-600 shadow-md'
                  : 'text-white/90 hover:text-white hover:bg-primary-600/80'
                  }`}
                style={{ animationDelay: `${index * 100}ms` }}
              >
                <span className="flex items-center space-x-2">
                  <span>{item.label}</span>
                </span>
                {location.pathname === item.path && (
                  <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-secondary-500 rounded-full animate-slide-up"></div>
                )}
              </Link>
            ))}
          </nav>

          {/* Desktop User Menu */}
          <div className="hidden md:flex items-center space-x-3">
            <div className="relative">
              <button
                onClick={toggleNotifications}
                className="relative p-2.5 text-white hover:text-secondary-500 hover:bg-primary-600 rounded-xl transition-all duration-300 transform hover:scale-110 group"
                title="Notifications"
              >
                <Bell className="h-5 w-5 transition-transform duration-300 group-hover:rotate-12" />
                {unreadCount > 0 && (
                  <span className="absolute top-1 right-1 w-2.5 h-2.5 bg-secondary-500 rounded-full animate-pulse border-2 border-primary-500"></span>
                )}
              </button>

              {/* Notifications Dropdown */}
              {showNotifications && (
                <div className="absolute right-0 mt-2 w-80 bg-white rounded-xl shadow-xl z-50 overflow-hidden border border-gray-100 animate-fade-in-down origin-top-right">
                  <div className="bg-primary-50 px-4 py-3 border-b border-primary-100 flex justify-between items-center">
                    <h3 className="text-sm font-semibold text-primary-800">Notifications</h3>
                    <span className="text-xs text-primary-600 bg-primary-100 px-2 py-0.5 rounded-full">{notifications.length} Total</span>
                  </div>
                  <div className="max-h-96 overflow-y-auto">
                    {notifications.length > 0 ? (
                      notifications.map((notification) => (
                        <div
                          key={notification.id}
                          className={`p-4 border-b border-gray-50 hover:bg-gray-50 transition-colors cursor-pointer ${!notification.is_read ? 'bg-blue-50/50' : ''}`}
                          onClick={() => handleNotificationClick(notification)}
                        >
                          <div className="flex items-start gap-3">
                            <div className={`mt-1 p-1.5 rounded-full flex-shrink-0 ${notification.type === 'appointment_request' ? 'bg-purple-100 text-purple-600' : 'bg-blue-100 text-blue-600'}`}>
                              {notification.type === 'appointment_request' ? <User className="h-3 w-3" /> : <Bell className="h-3 w-3" />}
                            </div>
                            <div>
                              <p className="text-sm text-gray-800 leading-snug">{notification.message}</p>
                              <p className="text-xs text-gray-400 mt-1">{new Date(notification.created_at).toLocaleDateString()} {new Date(notification.created_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</p>
                            </div>
                          </div>
                        </div>
                      ))
                    ) : (
                      <div className="p-8 text-center text-gray-500">
                        <Bell className="h-8 w-8 mx-auto mb-2 text-gray-300" />
                        <p className="text-sm">No notifications yet</p>
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>

            <div className="flex items-center space-x-3 px-4 py-2.5 bg-primary-600 rounded-xl shadow-md border border-primary-400/30 hover:bg-primary-700 transition-all duration-300 transform hover:scale-105">
              <div className="flex items-center space-x-2.5">
                <div className="bg-white/20 rounded-lg p-1.5">
                  {user?.role === 'doctor' ? (
                    <Stethoscope className="h-4 w-4 text-white" />
                  ) : (
                    <User className="h-4 w-4 text-white" />
                  )}
                </div>
                <div>
                  <p className="text-sm font-semibold text-white">{user?.username}</p>
                  <p className="text-xs text-white/80 capitalize font-medium">{user?.role}</p>
                </div>
              </div>
            </div>

            <button
              onClick={handleLogout}
              className="p-2.5 text-white hover:text-secondary-500 hover:bg-primary-600 rounded-xl transition-all duration-300 transform hover:scale-110 group"
              title="Logout"
            >
              <LogOut className="h-5 w-5 transition-transform duration-300 group-hover:-rotate-12" />
            </button>
          </div>

          {/* Mobile Menu Button */}
          <button
            onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
            className="md:hidden p-2.5 text-white hover:text-secondary-500 hover:bg-primary-600 rounded-xl transition-all duration-300 transform hover:scale-110 relative z-50"
            aria-label="Toggle menu"
          >
            <div className="relative w-6 h-6">
              {mobileMenuOpen ? (
                <X className="h-6 w-6 absolute inset-0 animate-rotate-in" />
              ) : (
                <Menu className="h-6 w-6 absolute inset-0" />
              )}
            </div>
          </button>
        </div>

        {/* Mobile Menu */}
        <div
          className={`md:hidden overflow-hidden transition-all duration-500 ease-in-out ${mobileMenuOpen
            ? 'max-h-[600px] opacity-100'
            : 'max-h-0 opacity-0'
            }`}
        >
          <div className="py-4 border-t-2 border-primary-400/30 bg-primary-600/50 backdrop-blur-md shadow-2xl rounded-b-2xl">
            <nav className="flex flex-col space-y-1 px-2">
              {getNavItems().map((item, index) => (
                <Link
                  key={item.path}
                  to={item.path}
                  onClick={() => setMobileMenuOpen(false)}
                  className={`px-4 py-3.5 rounded-xl text-sm font-semibold transition-all duration-300 transform hover:scale-[1.02] ${location.pathname === item.path
                    ? 'text-white bg-primary-700 shadow-lg border-2 border-secondary-500/50'
                    : 'text-white/90 hover:text-white hover:bg-primary-600/80'
                    }`}
                  style={{
                    animationDelay: `${index * 50}ms`,
                    animation: mobileMenuOpen ? 'slideInLeft 0.3s ease-out forwards' : 'none',
                  }}
                >
                  <span className="flex items-center space-x-3">
                    <span>{item.label}</span>
                    {location.pathname === item.path && (
                      <span className="ml-auto w-2 h-2 bg-secondary-500 rounded-full animate-pulse"></span>
                    )}
                  </span>
                </Link>
              ))}
              <div className="pt-4 mt-4 border-t-2 border-primary-400/30 space-y-2">
                <div className="px-4 py-3 flex items-center space-x-3 bg-primary-700/50 rounded-xl border border-primary-500/30">
                  <div className="bg-white/20 rounded-lg p-2">
                    {user?.role === 'doctor' ? (
                      <Stethoscope className="h-5 w-5 text-white" />
                    ) : (
                      <User className="h-5 w-5 text-white" />
                    )}
                  </div>
                  <div>
                    <p className="text-sm font-bold text-white">{user?.username}</p>
                    <p className="text-xs text-white/80 capitalize">{user?.role}</p>
                  </div>
                </div>
                <button
                  onClick={handleLogout}
                  className="w-full px-4 py-3.5 rounded-xl text-sm font-bold text-white bg-primary-700 hover:bg-primary-600 border-2 border-primary-500/50 hover:border-secondary-500/50 transition-all duration-300 transform hover:scale-[1.02] flex items-center justify-center space-x-2 shadow-lg"
                >
                  <LogOut className="h-5 w-5" />
                  <span>Logout</span>
                </button>
              </div>
            </nav>
          </div>
        </div>
      </div>
    </header>
  );
};

export default Header;
