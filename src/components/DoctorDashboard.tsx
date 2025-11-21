import React, { useState, useEffect } from 'react';
import { Users, FileText, Calendar, Activity, Eye, Download, Search, RefreshCcw } from 'lucide-react'; // Added RefreshCcw for refresh button
import { useAuth } from '../context/AuthContext';
import Header from './Header';

interface Report {
  id: number; // shared_report_id
  report_id: number; // original report ID
  patient_name: string;
  patient_email: string;
  date: string;
  result: string;
  confidence: string;
  status: 'new' | 'reviewed' | 'urgent';
  filename: string; // The original filename from the /uploads folder
}

interface DoctorAppointment {
  id: number;
  date: string; // YYYY-MM-DD
  time: string; // HH:mm or similar
  reason: string;
  status: string;
  patient_name: string;
  patient_email: string;
}

interface DoctorPatient {
  id: number;
  name: string;
  email: string;
  last_report_date: string | null;
  last_appointment_date: string | null; // might be combined date+time string
  total_reports_shared: number;
  total_appointments: number;
}

const DoctorDashboard = () => {
  const { user } = useAuth();
  const [activeTab, setActiveTab] = useState('shared-reports');
  const [sharedReports, setSharedReports] = useState<Report[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [searchTerm, setSearchTerm] = useState('');
  const [doctorAppointments, setDoctorAppointments] = useState<DoctorAppointment[]>([]);
  const [patients, setPatients] = useState<DoctorPatient[]>([]);

  const fetchSharedReports = async () => {
    setLoading(true);
    setError('');
    try {
      const response = await fetch('/api/shared-reports', {
        method: 'GET',
        credentials: 'include',
      });
      if (response.ok) {
        const data: Report[] = await response.json();
        setSharedReports(data);
      } else {
        const errorData = await response.json();
        setError(errorData.message || 'Failed to load shared reports');
      }
    } catch (err) {
      console.error('Error fetching shared reports:', err);
      setError('Error fetching shared reports. Please check your connection.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (activeTab === 'shared-reports') {
      fetchSharedReports();
    }
  }, [activeTab]);

  const fetchDoctorAppointments = async () => {
    setLoading(true);
    setError('');
    try {
      const response = await fetch('/api/doctor-appointments', {
        method: 'GET',
        credentials: 'include',
      });
      if (response.ok) {
        const data: DoctorAppointment[] = await response.json();
        setDoctorAppointments(data);
      } else {
        const errorData = await response.json();
        setError(errorData.message || 'Failed to load appointments');
      }
    } catch (err) {
      console.error('Error fetching doctor appointments:', err);
      setError('Error fetching appointments. Please check your connection.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (activeTab === 'appointments') {
      fetchDoctorAppointments();
    }
  }, [activeTab]);

  const updateAppointmentStatus = async (appointmentId: number, status: 'accepted' | 'declined' | 'pending') => {
    try {
      const response = await fetch(`/api/doctor-appointments/${appointmentId}/status`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify({ status }),
      });
      if (response.ok) {
        setDoctorAppointments((prev) => prev.map((a) => a.id === appointmentId ? { ...a, status } : a));
      } else {
        const errorData = await response.json();
        setError(errorData.message || 'Failed to update appointment status');
      }
    } catch (err) {
      console.error('Error updating appointment status:', err);
      setError('Error updating appointment status.');
    }
  };

  const fetchDoctorPatients = async () => {
    setLoading(true);
    setError('');
    try {
      const response = await fetch('/api/doctor-patients', {
        method: 'GET',
        credentials: 'include',
      });
      if (response.ok) {
        const data: DoctorPatient[] = await response.json();
        setPatients(data);
      } else {
        const errorData = await response.json();
        setError(errorData.message || 'Failed to load patients');
      }
    } catch (err) {
      console.error('Error fetching doctor patients:', err);
      setError('Error fetching patients. Please check your connection.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (activeTab === 'patient-management') {
      fetchDoctorPatients();
    }
  }, [activeTab]);

  const handleReview = async (sharedReportId: number, currentStatus: string) => {
    const newStatus = currentStatus === 'new' ? 'reviewed' : 'new'; // Toggle status
    try {
      const response = await fetch(`/api/shared-reports/${sharedReportId}/status`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify({ status: newStatus }),
      });
      if (response.ok) {
        // Update the local state to reflect the status change
        setSharedReports((prevReports) =>
          prevReports.map((report) =>
            report.id === sharedReportId ? { ...report, status: newStatus as 'new' | 'reviewed' | 'urgent' } : report
          )
        );
      } else {
        const errorData = await response.json();
        console.error('Failed to update report status:', errorData.message);
        // alert('Failed to update report status'); // Replacing alert
      }
    } catch (err) {
      console.error('Error updating report status:', err);
      // alert('Error updating report status'); // Replacing alert
    }
  };

  const handleDownload = async (reportId: number) => {
    try {
      // Use the actual report_id from the original report, not the shared_report_id
      const response = await fetch(`/api/download-report/${reportId}`, {
        method: 'GET',
        credentials: 'include',
      });
      if (response.ok) {
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `report_${reportId}.pdf`;
        document.body.appendChild(a); // Append to body to make it clickable in all browsers
        a.click();
        document.body.removeChild(a); // Clean up
        window.URL.revokeObjectURL(url);
      } else {
        const errorData = await response.json();
        console.error('Failed to download report:', errorData.message);
        // alert('Failed to download report'); // Replacing alert
      }
    } catch (err) {
      console.error('Error downloading report:', err);
      // alert('Error downloading report'); // Replacing alert
    }
  };

  const getResultColor = (result: string) => {
    if (result === 'None' || result === 'Not a fundus image') {
      return 'text-gray-600 bg-gray-50 border-gray-200';
    }
    const shortResult = result.split(' - ')[0];
    switch (shortResult) {
      case 'No DR': return 'text-green-600 bg-green-50 border-green-200';
      case 'Mild DR': return 'text-yellow-600 bg-yellow-50 border-yellow-200';
      case 'Moderate DR': return 'text-orange-600 bg-orange-50 border-orange-200';
      case 'Severe DR': return 'text-red-600 bg-red-50 border-red-200';
      case 'Proliferative DR': return 'text-purple-600 bg-purple-50 border-purple-200';
      default: return 'text-gray-600 bg-gray-50 border-gray-200';
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'new': return 'bg-blue-100 text-blue-700 border-blue-200';
      case 'reviewed': return 'bg-gray-100 text-gray-700 border-gray-200';
      case 'urgent': return 'bg-red-100 text-red-700 border-red-200';
      default: return 'bg-gray-100 text-gray-700 border-gray-200';
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-background-100 flex items-center justify-center">
        <div className="flex items-center space-x-2 text-primary-700">
          <RefreshCcw className="h-5 w-5 animate-spin" />
          <span>Loading...</span>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-background-100 flex items-center justify-center">
        <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded-lg relative" role="alert">
          <strong className="font-bold">Error!</strong>
          <span className="block sm:inline"> {error}</span>
        </div>
      </div>
    );
  }

  const filteredReports = sharedReports.filter((report) =>
    report.patient_name.toLowerCase().includes(searchTerm.toLowerCase()) ||
    report.patient_email.toLowerCase().includes(searchTerm.toLowerCase())
  );

  const totalSharedReports = sharedReports.length;
  const uniquePatients = new Set(sharedReports.map((r) => r.patient_email)).size;
  const pendingReviews = sharedReports.filter((r) => r.status === 'new').length;
  const upcomingAppointments = doctorAppointments.filter((a) => {
    const dateTimeIso = `${a.date}T${a.time}`;
    const dt = new Date(dateTimeIso);
    return !Number.isNaN(dt.getTime()) && dt.getTime() > Date.now();
  }).length;

  return (
    <div className="min-h-screen bg-background-100">
      <Header />
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-primary-900 mb-2">
            Welcome Dr. {user?.username}
          </h1>
          <p className="text-primary-600">
            Review patient reports and manage care with AI-powered insights.
          </p>
        </div>

        <div className="grid lg:grid-cols-4 gap-6 mb-8">
          <div className="bg-white rounded-xl p-6 shadow-sm border border-primary-100">
            <div className="flex items-center justify-between mb-4">
              <div className="bg-blue-500 rounded-lg p-3">
                <FileText className="h-6 w-6 text-white" />
              </div>
              <span className="bg-blue-100 text-blue-600 px-2 py-1 rounded-full text-xs font-medium">
                +{sharedReports.filter((r) => new Date(r.date) > new Date(Date.now() - 24 * 60 * 60 * 1000)).length} today
              </span>
            </div>
            <h3 className="text-2xl font-bold text-primary-900">{totalSharedReports}</h3>
            <p className="text-sm text-primary-600">Shared Reports</p>
          </div>
          <div className="bg-white rounded-xl p-6 shadow-sm border border-primary-100">
            <div className="flex items-center justify-between mb-4">
              <div className="bg-primary-500 rounded-lg p-3">
                <Users className="h-6 w-6 text-white" />
              </div>
            </div>
            <h3 className="text-2xl font-bold text-primary-900">
              {uniquePatients}
            </h3>
            <p className="text-sm text-primary-600">Unique Patients</p>
          </div>
          <div className="bg-white rounded-xl p-6 shadow-sm border border-primary-100">
            <div className="flex items-center justify-between mb-4">
              <div className="bg-gold-500 rounded-lg p-3">
                <Activity className="h-6 w-6 text-white" />
              </div>
            </div>
            <h3 className="text-2xl font-bold text-primary-900">
              {pendingReviews}
            </h3>
            <p className="text-sm text-primary-600">Pending Reviews</p>
          </div>
          <div className="bg-white rounded-xl p-6 shadow-sm border primary-100">
            <div className="flex items-center justify-between mb-4">
              <div className="bg-purple-500 rounded-lg p-3">
                <Calendar className="h-6 w-6 text-white" />
              </div>
            </div>
            <h3 className="text-2xl font-bold text-primary-900">{upcomingAppointments}</h3>
            <p className="text-sm text-primary-600">Upcoming Appointments</p>
          </div>
        </div>

        <div className="mb-6">
          <div className="flex space-x-4 border-b border-primary-100">
            <button
              onClick={() => setActiveTab('shared-reports')}
              className={`px-4 py-2 font-medium text-sm transition-colors ${
                activeTab === 'shared-reports'
                  ? 'text-primary-700 border-b-2 border-primary-600'
                  : 'text-primary-600 hover:text-primary-700'
              }`}
            >
              Shared Reports
            </button>
            <button
              onClick={() => setActiveTab('patient-management')}
              className={`px-4 py-2 font-medium text-sm transition-colors ${
                activeTab === 'patient-management'
                  ? 'text-primary-700 border-b-2 border-primary-600'
                  : 'text-primary-600 hover:text-primary-700'
              }`}
            >
              Patient Management
            </button>
            <button
              onClick={() => setActiveTab('appointments')}
              className={`px-4 py-2 font-medium text-sm transition-colors ${
                activeTab === 'appointments'
                  ? 'text-primary-700 border-b-2 border-primary-600'
                  : 'text-primary-600 hover:text-primary-700'
              }`}
            >
              Appointments
            </button>
          </div>
        </div>

        <div className="bg-white rounded-2xl shadow-sm border border-primary-100 p-6">
          {activeTab === 'shared-reports' && (
            <>
              <div className="flex items-center justify-between mb-6">
                <h2 className="text-xl font-semibold text-primary-900">Patient Reports</h2>
                <div className="relative">
                  <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400" />
                  <input
                    type="text"
                    placeholder="Search patients..."
                    value={searchTerm}
                    onChange={(e) => setSearchTerm(e.target.value)}
                    className="pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                  />
                </div>
              </div>

              <div className="space-y-4">
                {filteredReports.length > 0 ? (
                  filteredReports.map((report) => (
                    <div key={report.id} className="flex flex-col sm:flex-row items-start sm:items-center justify-between p-4 bg-primary-50 rounded-lg hover:bg-primary-100 transition-colors">
                      <div className="flex items-start space-x-4 mb-3 sm:mb-0">
                        <div className="bg-primary-100 rounded-lg p-2">
                          <Eye className="h-5 w-5 text-primary-600" />
                        </div>
                        <div>
                          <h3 className="font-semibold text-primary-900">{report.patient_name}</h3>
                          <p className="text-sm text-primary-600">{report.patient_email}</p>
                          <p className="text-xs text-gray-500">{report.date}</p>
                        </div>
                      </div>
                      <div className="flex flex-col sm:flex-row items-end sm:items-center space-y-2 sm:space-y-0 sm:space-x-4 w-full sm:w-auto">
                        <div className="text-right">
                          <span className={`px-3 py-1 rounded-full text-xs font-medium border ${getResultColor(report.result)}`}>
                            {report.result}
                          </span>
                          <p className="text-xs text-gray-500 mt-1">Confidence: {report.confidence}</p>
                        </div>
                        <span className={`px-2 py-1 rounded-full text-xs font-medium border ${getStatusColor(report.status)}`}>
                          {report.status}
                        </span>
                        <div className="flex items-center space-x-2">
                          <button
                            onClick={() => handleReview(report.id, report.status)}
                            className="text-primary-600 hover:text-primary-800 text-sm font-medium"
                          >
                            {report.status === 'new' ? 'Mark as Reviewed' : 'Mark as New'}
                          </button>
                          <button
                            onClick={() => handleDownload(report.report_id)}
                            className="text-gray-500 hover:text-gray-700"
                            title="Download Report"
                          >
                            <Download className="h-4 w-4" />
                          </button>
                        </div>
                      </div>
                    </div>
                  ))
                ) : (
                  <div className="text-center py-8 text-gray-500">
                    No reports shared with you yet.
                  </div>
                )}
              </div>
            </>
          )}

          {activeTab === 'patient-management' && (
            <>
              <div className="flex items-center justify-between mb-6">
                <h2 className="text-xl font-semibold text-primary-900">Patient Management</h2>
                <button
                  onClick={fetchDoctorPatients}
                  className="inline-flex items-center gap-2 text-sm text-primary-700 hover:text-primary-900"
                >
                  <RefreshCcw className="h-4 w-4" /> Refresh
                </button>
              </div>
              <div className="space-y-4">
                {patients.length > 0 ? (
                  patients.map((p) => (
                    <div key={p.id} className="flex flex-col sm:flex-row items-start sm:items-center justify-between p-4 bg-primary-50 rounded-lg">
                      <div className="flex items-start space-x-4 mb-3 sm:mb-0">
                        <div className="bg-primary-100 rounded-lg p-2">
                          <Users className="h-5 w-5 text-primary-600" />
                        </div>
                        <div>
                          <h3 className="font-semibold text-primary-900">{p.name}</h3>
                          <p className="text-sm text-primary-600">{p.email}</p>
                          <div className="text-xs text-gray-600 mt-1">
                            {p.last_report_date && <span className="mr-3">Last Report: {p.last_report_date}</span>}
                            {p.last_appointment_date && <span>Last Appointment: {p.last_appointment_date}</span>}
                          </div>
                        </div>
                      </div>
                      <div className="flex items-center gap-3 w-full sm:w-auto">
                        <span className="px-3 py-1 rounded-full text-xs font-medium border bg-gray-50 text-gray-700 border-gray-200">
                          {p.total_reports_shared} reports
                        </span>
                        <span className="px-3 py-1 rounded-full text-xs font-medium border bg-gray-50 text-gray-700 border-gray-200">
                          {p.total_appointments} appointments
                        </span>
                      </div>
                    </div>
                  ))
                ) : (
                  <div className="text-center py-8 text-gray-500">No patients yet.</div>
                )}
              </div>
            </>
          )}

          {activeTab === 'appointments' && (
            <>
              <div className="flex items-center justify-between mb-6">
                <h2 className="text-xl font-semibold text-primary-900">Appointments</h2>
                <button
                  onClick={fetchDoctorAppointments}
                  className="inline-flex items-center gap-2 text-sm text-primary-700 hover:text-primary-900"
                >
                  <RefreshCcw className="h-4 w-4" /> Refresh
                </button>
              </div>

              <div className="space-y-4">
                {doctorAppointments.length > 0 ? (
                  doctorAppointments.map((appt) => (
                    <div key={appt.id} className="flex flex-col sm:flex-row items-start sm:items-center justify-between p-4 bg-primary-50 rounded-lg">
                      <div className="flex items-start space-x-4 mb-3 sm:mb-0">
                        <div className="bg-purple-100 rounded-lg p-2">
                          <Calendar className="h-5 w-5 text-purple-600" />
                        </div>
                        <div>
                          <h3 className="font-semibold text-primary-900">{appt.patient_name}</h3>
                          <p className="text-sm text-primary-600">{appt.patient_email}</p>
                          <p className="text-xs text-gray-500">{appt.date} at {appt.time}</p>
                          {appt.reason && (
                            <p className="text-xs text-gray-600 mt-1">Reason: {appt.reason}</p>
                          )}
                        </div>
                      </div>
                      <div className="flex items-center gap-3 w-full sm:w-auto">
                        <span className="px-3 py-1 rounded-full text-xs font-medium border bg-gray-50 text-gray-700 border-gray-200">
                          {appt.status}
                        </span>
                        {appt.status !== 'accepted' && (
                          <button
                            onClick={() => updateAppointmentStatus(appt.id, 'accepted')}
                            className="text-green-600 hover:text-green-800 text-sm font-medium"
                          >
                            Accept
                          </button>
                        )}
                        {appt.status !== 'declined' && (
                          <button
                            onClick={() => updateAppointmentStatus(appt.id, 'declined')}
                            className="text-red-600 hover:text-red-800 text-sm font-medium"
                          >
                            Decline
                          </button>
                        )}
                      </div>
                    </div>
                  ))
                ) : (
                  <div className="text-center py-8 text-gray-500">No appointments found.</div>
                )}
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
};

export default DoctorDashboard;
