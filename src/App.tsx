import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import LandingPage from './components/LandingPage';
import Login from './components/Login';
import Register from './components/Register';
import PatientDashboard from './components/PatientDashboard';
import DoctorDashboard from './components/DoctorDashboard';
import Predict from './components/Predict';
import Results from './components/Results';
import ResultDetail from './components/ResultDetail';
import Appointments from './components/Appointments';
import NewAnalysis from './components/NewAnalysis';
import ClinicalPredict from './components/ClinicalPredict';
import FloatingChatbot from './components/FloatingChatbot';
import { AuthProvider, useAuth } from './context/AuthContext';

// Enhanced Protected Route Component
interface ProtectedRouteProps {
  children: React.ReactNode;
  requiredRole?: string;
  redirectTo?: string;
}

function ProtectedRoute({ children, requiredRole, redirectTo = '/login' }: ProtectedRouteProps) {
  const { user } = useAuth();
  
  if (!user) {
    return <Navigate to={redirectTo} replace />;
  }
  
  if (requiredRole && user.role !== requiredRole) {
    return <Navigate to={`/${user.role}-dashboard`} replace />;
  }
  
  return <>{children}</>;
}

function AppRoutes() {
  const { user } = useAuth();

  return (
    <Routes>
      <Route path="/" element={<LandingPage />} />
      <Route 
        path="/login" 
        element={user ? <Navigate to={`/${user.role}-dashboard`} replace /> : <Login />} 
      />
      <Route 
        path="/register" 
        element={user ? <Navigate to={`/${user.role}-dashboard`} replace /> : <Register />} 
      />
      
      {/* Protected Patient Routes */}
      <Route 
        path="/patient-dashboard" 
        element={
          <ProtectedRoute requiredRole="patient">
            <PatientDashboard />
          </ProtectedRoute>
        } 
      />
      <Route 
        path="/new-analysis" 
        element={
          <ProtectedRoute requiredRole="patient">
            <NewAnalysis />
          </ProtectedRoute>
        } 
      />
      <Route 
        path="/predict" 
        element={
          <ProtectedRoute requiredRole="patient">
            <Predict />
          </ProtectedRoute>
        } 
      />
      <Route 
        path="/clinical-predict" 
        element={
          <ProtectedRoute requiredRole="patient">
            <ClinicalPredict />
          </ProtectedRoute>
        } 
      />
      <Route 
        path="/results" 
        element={
          <ProtectedRoute requiredRole="patient">
            <Results />
          </ProtectedRoute>
        } 
      />
      <Route 
        path="/result/:id" 
        element={
          <ProtectedRoute requiredRole="patient">
            <ResultDetail />
          </ProtectedRoute>
        } 
      />
      <Route 
        path="/appointments" 
        element={
          <ProtectedRoute requiredRole="patient">
            <Appointments />
          </ProtectedRoute>
        } 
      />
      
      {/* Protected Doctor Routes */}
      <Route 
        path="/doctor-dashboard" 
        element={
          <ProtectedRoute requiredRole="doctor">
            <DoctorDashboard />
          </ProtectedRoute>
        } 
      />
      
      {/* Fallback */}
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}

function App() {
  return (
    <AuthProvider>
      <Router>
        <div className="min-h-screen bg-background-100">
          <AppRoutes />
          <FloatingChatbot />
        </div>
      </Router>
    </AuthProvider>
  );
}

export default App;