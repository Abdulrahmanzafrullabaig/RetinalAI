/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        primary: {
          50: '#E6F2F6',
          100: '#CCE5ED',
          200: '#99CBDB',
          300: '#66B1C9',
          400: '#3397B7',
          500: '#2582A1', // Apollo Teal Blue - Main Brand Color
          600: '#1F728F', // Hover State
          700: '#1A627D',
          800: '#14526B',
          900: '#0E4259',
          DEFAULT: '#2582A1',
        },
        secondary: {
          50: '#FFF9E6',
          100: '#FFF3CC',
          200: '#FFE799',
          300: '#FFDB66',
          400: '#FFCF33',
          500: '#FDB931', // Apollo Saffron Yellow - CTA/Accent
          600: '#E6A72C', // Hover State
          700: '#CC9527',
          800: '#B38322',
          900: '#99711D',
          DEFAULT: '#FDB931',
        },
        // Mapped 'gold' to saffron for backward compatibility
        gold: {
          400: '#FDB931', // Saffron Yellow
          500: '#FDB931', // Saffron Yellow
          600: '#E6A72C', // Hover
        },
        // Neutral Scale - Apollo inspired
        gray: {
          50: '#FFFFFF', // White background
          100: '#F7F9FA', // Light background
          200: '#EEF3F6', // Card Background
          300: '#D3D8DE', // Borders/Dividers (Apollo specified)
          400: '#B0BCC5',
          500: '#6C757D', // Secondary Text (Apollo specified)
          600: '#5A6268',
          700: '#4A5A67', // Sub-headings
          800: '#34424D',
          900: '#1E2A32', // Primary Text (Apollo specified)
        },
        // Semantic mappings
        background: {
          DEFAULT: '#FFFFFF', // White (Apollo specified)
          paper: '#FFFFFF',   // Card standard
          100: '#F7F9FA',    // Light background variant
        },
        text: {
          main: '#1E2A32', // Primary Text (Apollo specified)
          sub: '#6C757D',  // Secondary Text (Apollo specified)
          secondary: '#6C757D',
        },
        border: {
          DEFAULT: '#D3D8DE', // Borders/Dividers (Apollo specified)
          focus: '#2582A1',   // Focus state uses primary teal
        },
        // Accents
        sky: '#8ECFFF',
        lime: '#A6F7C5',
        red: {
          DEFAULT: '#FF6B6B', // Soft Red
          50: '#FFF0F0',
          500: '#FF6B6B',
          600: '#E55F5F',
        },
        amber: {
          DEFAULT: '#FDB931', // Saffron Yellow
          500: '#FDB931',
          600: '#E6A72C',
        },
        // Backwards compatibility mappings if needed
        charcoal: {
          DEFAULT: '#1E2A32',
          500: '#1E2A32',
          600: '#1E2A32',
          700: '#4A5A67',
          900: '#1E2A32',
        }
      },
      fontFamily: {
        'heading': ['Playfair Display', 'serif'],
        'body': ['Roboto', 'sans-serif'],
        'sans': ['Roboto', 'system-ui', 'sans-serif'],
        'serif': ['Playfair Display', 'serif'],
      },
      spacing: {
        '18': '4.5rem',
        '88': '22rem',
      },
      animation: {
        'fade-in': 'fadeIn 0.6s ease-out',
        'fade-in-up': 'fadeInUp 0.6s ease-out',
        'fade-in-down': 'fadeInDown 0.6s ease-out',
        'slide-up': 'slideUp 0.5s ease-out',
        'slide-down': 'slideDown 0.5s ease-out',
        'slide-in-left': 'slideInLeft 0.5s ease-out',
        'slide-in-right': 'slideInRight 0.5s ease-out',
        'scale-in': 'scaleIn 0.4s ease-out',
        'bounce-in': 'bounceIn 0.8s ease-out',
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'float': 'float 3s ease-in-out infinite',
        'shimmer': 'shimmer 2s linear infinite',
        'rotate-in': 'rotateIn 0.3s ease-out',
      },
      keyframes: {
        fadeIn: {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
        fadeInUp: {
          '0%': { opacity: '0', transform: 'translateY(20px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        fadeInDown: {
          '0%': { opacity: '0', transform: 'translateY(-20px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        slideUp: {
          '0%': { transform: 'translateY(30px)', opacity: '0' },
          '100%': { transform: 'translateY(0)', opacity: '1' },
        },
        slideDown: {
          '0%': { transform: 'translateY(-30px)', opacity: '0' },
          '100%': { transform: 'translateY(0)', opacity: '1' },
        },
        slideInLeft: {
          '0%': { transform: 'translateX(-30px)', opacity: '0' },
          '100%': { transform: 'translateX(0)', opacity: '1' },
        },
        slideInRight: {
          '0%': { transform: 'translateX(30px)', opacity: '0' },
          '100%': { transform: 'translateX(0)', opacity: '1' },
        },
        scaleIn: {
          '0%': { transform: 'scale(0.9)', opacity: '0' },
          '100%': { transform: 'scale(1)', opacity: '1' },
        },
        bounceIn: {
          '0%': { transform: 'scale(0.3)', opacity: '0' },
          '50%': { transform: 'scale(1.05)' },
          '70%': { transform: 'scale(0.9)' },
          '100%': { transform: 'scale(1)', opacity: '1' },
        },
        float: {
          '0%, 100%': { transform: 'translateY(0px)' },
          '50%': { transform: 'translateY(-10px)' },
        },
        shimmer: {
          '0%': { backgroundPosition: '-1000px 0' },
          '100%': { backgroundPosition: '1000px 0' },
        },
        rotateIn: {
          '0%': { transform: 'rotate(-90deg)', opacity: '0' },
          '100%': { transform: 'rotate(0deg)', opacity: '1' },
        },
      },
      boxShadow: {
        'soft': '0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03)',
        'glow': '0 0 20px rgba(37, 130, 161, 0.25)', // Apollo teal glow
        'glow-yellow': '0 0 20px rgba(253, 185, 49, 0.25)', // Apollo saffron glow
        'card': '0 2px 8px rgba(0, 0, 0, 0.08)',
        'card-hover': '0 8px 24px rgba(0, 0, 0, 0.12)',
      },
      transitionDuration: {
        '400': '400ms',
        '600': '600ms',
        '800': '800ms',
      }
    },
  },
  plugins: [],
};