/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        // Primary Color: Charcoal (#334155)
        primary: {
          50: '#F5F6F7',
          100: '#E8EBED',
          200: '#D1D7DB',
          300: '#BAC3C9',
          400: '#A3AFB7',
          500: '#334155', // Primary color - Charcoal
          600: '#293749',
          700: '#1F2D3D',
          800: '#152331',
          900: '#0B1925',
        },
        // Background Color: Light Cool Gray (#F1F5F9)
        background: {
          50: '#FFFFFF',
          100: '#F1F5F9', // Background color - Light Cool Gray
          200: '#E2E8F0',
          300: '#CBD5E1',
          400: '#94A3B8',
          500: '#64748B',
          600: '#475569', // Body text color
          700: '#334155',
          800: '#1E293B',
          900: '#0F172A',
        },
        // Highlight/CTA Color: Warm Gold (#F59E0B)
        gold: {
          50: '#FFFBEB',
          100: '#FEF3C7',
          200: '#FDE68A',
          300: '#FCD34D',
          400: '#FBBF24',
          500: '#F59E0B', // CTA color - Warm Gold
          600: '#D97706',
          700: '#B45309',
          800: '#92400E',
          900: '#78350F',
        },
        // Keep existing teal for backward compatibility
        teal: {
          50: '#E0F2F1',
          100: '#B2DFDB',
          200: '#80CBC4',
          300: '#4DB6AC',
          400: '#26A69A',
          500: '#16A085',
          600: '#16A085',
          700: '#00796B',
          800: '#00695C',
          900: '#004D40',
        },
        // Keep coral for backward compatibility
        coral: {
          50: '#FEF2F0',
          100: '#FDE5E0',
          200: '#FBC4B8',
          300: '#F9A390',
          400: '#F68268',
          500: '#F2784B',
          600: '#E55A2B',
          700: '#C2441F',
          800: '#9F3015',
          900: '#7C1E0B',
        }
      },
      fontFamily: {
        'heading': ['Playfair Display', 'serif'], // Elegant serif for headings
        'body': ['Roboto', 'sans-serif'], // Clean sans-serif for body text
        'sans': ['Roboto', 'system-ui', 'sans-serif'], // Override default sans
        'serif': ['Playfair Display', 'serif'], // Override default serif
      },
      spacing: {
        '18': '4.5rem',
        '88': '22rem',
      },
      animation: {
        'fade-in': 'fadeIn 0.5s ease-in-out',
        'slide-up': 'slideUp 0.3s ease-out',
      },
      keyframes: {
        fadeIn: {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
        slideUp: {
          '0%': { transform: 'translateY(10px)', opacity: '0' },
          '100%': { transform: 'translateY(0)', opacity: '1' },
        },
      },
    },
  },
  plugins: [],
};