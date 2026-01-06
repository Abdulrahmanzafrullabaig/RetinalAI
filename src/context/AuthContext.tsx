import React, { createContext, useContext, useState, useEffect } from 'react';
import { API_URL } from '../config';

// Define the User interface matching the backend's user data structure
interface User {
  id: number; // Assuming id is numeric from SQLite AUTOINCREMENT
  username: string;
  email: string;
  role: 'patient' | 'doctor';
  full_name?: string; // Optional, might be used in some places
}

interface AuthContextType {
  user: User | null;
  login: (email: string, password: string, role: 'patient' | 'doctor') => Promise<boolean>;
  register: (username: string, email: string, password: string, role: 'patient' | 'doctor', fullName?: string) => Promise<{ success: boolean; message?: string }>;
  logout: () => void;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<User | null>(null);

  // On initial load, check for a stored user session
  useEffect(() => {
    const checkAuthStatus = async () => {
      try {
        const response = await fetch(`${API_URL}/api/user`, {
          method: 'GET',
          credentials: 'include', // Important for sending session cookies
        });

        if (response.ok) {
          const data = await response.json();
          if (data.success && data.user) {
            setUser(data.user);
          } else {
            setUser(null); // Clear user if session is invalid
            localStorage.removeItem('user');
          }
        } else {
          // If not OK, user is not authenticated or session expired
          setUser(null);
          localStorage.removeItem('user');
        }
      } catch (error) {
        console.error('Error checking auth status:', error);
        setUser(null);
        localStorage.removeItem('user');
      }
    };

    checkAuthStatus();
  }, []);

  const login = async (email: string, password: string, role: 'patient' | 'doctor'): Promise<boolean> => {
    try {
      const response = await fetch(`${API_URL}/api/login`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        credentials: 'include', // Important for receiving session cookies
        body: JSON.stringify({ email, password, role }),
      });

      const data = await response.json();

      if (response.ok && data.success) {
        // Backend now sends the user object directly
        setUser(data.user);
        // localStorage.setItem('user', JSON.stringify(data.user)); // Session handled by Flask
        return true;
      } else {
        console.error('Login failed:', data.message);
        return false;
      }
    } catch (error) {
      console.error('Error during login:', error);
      return false;
    }
  };

  interface RegisterResponse {
    success: boolean;
    message?: string;
  }

  const register = async (username: string, email: string, password: string, role: 'patient' | 'doctor', fullName?: string): Promise<RegisterResponse> => {
    try {
      const response = await fetch(`${API_URL}/api/register`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        credentials: 'include',
        body: JSON.stringify({ username, email, password, role, full_name: fullName }),
      });

      const data = await response.json();

      if (response.ok && data.success) {
        console.log('Registration successful:', data.message);
        return { success: true, message: data.message };
      } else {
        console.error('Registration failed:', data.message);
        return { success: false, message: data.message || 'Registration failed' };
      }
    } catch (error) {
      console.error('Error during registration:', error);
      return { success: false, message: 'Network error during registration' };
    }
  };

  const logout = async () => {
    try {
      const response = await fetch(`${API_URL}/api/logout`, {
        method: 'POST',
        credentials: 'include', // Important for sending session cookies
      });

      if (response.ok) {
        setUser(null);
        // localStorage.removeItem('user'); // Session handled by Flask
        console.log('Logged out successfully.');
      } else {
        console.error('Logout failed:', await response.json());
      }
    } catch (error) {
      console.error('Error during logout:', error);
    }
  };

  return (
    <AuthContext.Provider value={{ user, login, register, logout }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
}
