import React, { useState, useEffect } from 'react';
import { Calendar, Clock, User, Plus, Phone, Mail, MapPin, RefreshCcw } from 'lucide-react';
import Header from './Header';

interface Doctor {
  id: number;
  name: string;
  email: string;
}

interface Appointment {
  id: number;
  doctor_name: string;
  date: string;
  time: string;
  reason: string;
  status: 'pending' | 'confirmed' | 'cancelled';
}

const Appointments = () => {
  const [showBooking, setShowBooking] = useState(false);
  const [formData, setFormData] = useState({
    doctor: '',
    date: '',
    time: '',
    reason: '',
  });
  const [appointments, setAppointments] = useState<Appointment[]>([]);
  const [availableDoctors, setAvailableDoctors] = useState<Doctor[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  const fetchData = async () => {
    setLoading(true);
    setError('');
    try {
      const appResponse = await fetch('/api/appointments', {
        method: 'GET',
        credentials: 'include',
      });
      if (appResponse.ok) {
        const appData: Appointment[] = await appResponse.json();
        setAppointments(appData);
      } else {
        const errorData = await appResponse.json();
        setError(errorData.message || 'Failed to load appointments');
      }

      const docResponse = await fetch('/api/doctors', {
        method: 'GET',
        credentials: 'include',
      });
      if (docResponse.ok) {
        const docData: Doctor[] = await docResponse.json();
        setAvailableDoctors(docData);
      } else {
        const errorData = await docResponse.json();
        setError((prev) => prev + (prev ? ' | ' : '') + (errorData.message || 'Failed to load doctors'));
      }
    } catch (err) {
      console.error('Error fetching data:', err);
      setError('Error fetching data. Please check your connection.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
  }, []);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!formData.doctor || !formData.date || !formData.time) {
      // alert('Please fill in all required fields'); // Replacing alert
      setError('Please fill in all required fields');
      return;
    }
    setError(''); // Clear previous errors
    try {
      const response = await fetch('/api/appointments', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify({
          doctor: formData.doctor,
          date: formData.date,
          time: formData.time,
          reason: formData.reason,
        }),
      });
      const data = await response.json();
      if (response.ok && data.success) {
        // Refetch appointments to show the newly booked one
        await fetchData();
        setShowBooking(false);
        setFormData({ doctor: '', date: '', time: '', reason: '' });
      } else {
        setError(data.message || 'Failed to book appointment');
      }
    } catch (err) {
      console.error('Error booking appointment:', err);
      setError('Error booking appointment. Please check your connection.');
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'confirmed': return 'bg-green-100 text-green-700 border-green-200';
      case 'pending': return 'bg-yellow-100 text-yellow-700 border-yellow-200';
      case 'cancelled': return 'bg-red-100 text-red-700 border-red-200'; // Fixed red-100 text-red-100
      default: return 'bg-gray-100 text-gray-700 border-gray-200';
    }
  };

  // Get today's date in YYYY-MM-DD format to disable past dates
  const getMinDate = () => {
    const today = new Date();
    const year = today.getFullYear();
    const month = (today.getMonth() + 1).toString().padStart(2, '0');
    const day = today.getDate().toString().padStart(2, '0');
    return `${year}-${month}-${day}`;
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-background-100 flex items-center justify-center">
        <div className="flex items-center space-x-2 text-primary-700">
          <RefreshCcw className="h-5 w-5 animate-spin" />
          <span>Loading Appointments...</span>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-background-100">
      <Header />
      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="flex justify-between items-center mb-8">
          <div>
            <h1 className="text-3xl font-bold text-primary-900 mb-2">Appointments</h1>
            <p className="text-primary-600">Manage your eye care appointments</p>
          </div>
          <button
            onClick={() => setShowBooking(true)}
            className="bg-gold-500 hover:bg-gold-600 text-white px-6 py-3 rounded-lg font-semibold transition-colors inline-flex items-center space-x-2"
          >
            <Plus className="h-5 w-5" />
            <span>Book Appointment</span>
          </button>
        </div>

        {error && (
          <div className="mb-4 bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-lg text-sm" role="alert">
            {error}
          </div>
        )}

        <div className="grid lg:grid-cols-3 gap-8">
          <div className="lg:col-span-2">
            <div className="bg-white rounded-2xl shadow-sm border border-primary-100 p-6">
              <h2 className="text-xl font-semibold text-primary-900 mb-6">Upcoming Appointments</h2>
              <div className="space-y-4">
                {appointments.length > 0 ? (
                  appointments.map((appointment) => (
                    <div key={appointment.id} className="border border-primary-100 rounded-xl p-4 hover:shadow-sm transition-shadow">
                      <div className="flex items-start justify-between mb-3">
                        <div className="flex items-start space-x-3">
                          <div className="bg-primary-100 rounded-lg p-2">
                            <User className="h-5 w-5 text-primary-600" />
                          </div>
                          <div>
                            <h3 className="font-semibold text-primary-900">{appointment.doctor_name}</h3>
                            <p className="text-sm text-primary-600 flex items-center space-x-2">
                              <Calendar className="h-4 w-4" />
                              <span>{appointment.date} at {appointment.time}</span>
                            </p>
                            <p className="text-sm text-primary-600">{appointment.reason || 'No specific reason provided'}</p>
                          </div>
                        </div>
                        <span className={`px-3 py-1 rounded-full text-xs font-medium border ${getStatusColor(appointment.status)}`}>
                          {appointment.status}
                        </span>
                      </div>
                    </div>
                  ))
                ) : (
                  <div className="text-center py-8 text-gray-500">
                    No appointments booked yet.
                  </div>
                )}
              </div>
            </div>
          </div>

          {showBooking && (
            <div className="lg:col-span-1">
              <div className="bg-white rounded-2xl shadow-sm border border-primary-100 p-6">
                <h2 className="text-xl font-semibold text-primary-900 mb-6">Book New Appointment</h2>
                <form onSubmit={handleSubmit} className="space-y-4">
                  <div>
                    <label htmlFor="doctor" className="block text-sm font-medium text-primary-700 mb-2">Doctor</label>
                    <select
                      id="doctor"
                      value={formData.doctor}
                      onChange={(e) => setFormData({ ...formData, doctor: e.target.value })}
                      className="w-full px-4 py-3 border border-primary-200 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                      required
                    >
                      <option value="">Select a doctor</option>
                      {availableDoctors.map((doctor) => (
                        <option key={doctor.id} value={doctor.id}>
                          {doctor.name} ({doctor.email})
                        </option>
                      ))}
                    </select>
                  </div>
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <label htmlFor="date" className="block text-sm font-medium text-primary-700 mb-2">Date</label>
                      <input
                        id="date"
                        type="date"
                        value={formData.date}
                        onChange={(e) => setFormData({ ...formData, date: e.target.value })}
                        className="w-full px-4 py-3 border border-primary-200 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                        min={getMinDate()} // Prevent selecting past dates
                        required
                      />
                    </div>
                    <div>
                      <label htmlFor="time" className="block text-sm font-medium text-primary-700 mb-2">Time</label>
                      <select
                        id="time"
                        value={formData.time}
                        onChange={(e) => setFormData({ ...formData, time: e.target.value })}
                        className="w-full px-4 py-3 border border-primary-200 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                        required
                      >
                        <option value="">Select time</option>
                        <option value="09:00">9:00 AM</option>
                        <option value="10:00">10:00 AM</option>
                        <option value="11:00">11:00 AM</option>
                        <option value="12:00">12:00 PM</option>
                        <option value="13:00">1:00 PM</option>
                        <option value="14:00">2:00 PM</option>
                        <option value="15:00">3:00 PM</option>
                        <option value="16:00">4:00 PM</option>
                        <option value="17:00">5:00 PM</option>
                      </select>
                    </div>
                  </div>
                  <div>
                    <label htmlFor="reason" className="block text-sm font-medium text-primary-700 mb-2">
                      Reason for Visit
                    </label>
                    <textarea
                      id="reason"
                      value={formData.reason}
                      onChange={(e) => setFormData({ ...formData, reason: e.target.value })}
                      className="w-full px-4 py-3 border border-primary-200 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                      rows={3}
                      placeholder="Brief description of your concern..."
                    />
                  </div>
                  <div className="flex space-x-3 pt-4">
                    <button
                      type="button"
                      onClick={() => {
                        setShowBooking(false);
                        setError(''); // Clear error on cancel
                      }}
                      className="flex-1 px-4 py-3 border border-gray-300 text-gray-700 rounded-lg hover:bg-gray-50 font-medium transition-colors"
                    >
                      Cancel
                    </button>
                    <button
                      type="submit"
                      className="flex-1 px-4 py-3 bg-primary-600 hover:bg-primary-700 text-white rounded-lg font-medium transition-colors"
                    >
                      Book Appointment
                    </button>
                  </div>
                </form>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default Appointments;
