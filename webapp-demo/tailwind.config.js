/** @type {import('tailwindcss').Config} */
import forms from '@tailwindcss/forms';

export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'everest': {
          '1000': '#0D1320',
          '800': '#112033',
        },
        'glacier': {
          '600': '#4A6FA5',
          '300': '#8FAFE0',
        },
        'aurora': {
          'green': '#16E0A2',
          'purple': '#B569FF',
        },
        'snow': '#EAF5FF',
      },
      backdropBlur: {
        'xs': '2px',
        'sm': '4px',
        'md': '8px',
        'lg': '12px',
        'xl': '16px',
      },
    },
    fontFamily: {
      'sans': ['Inter', 'Space Grotesk', 'ui-sans-serif', 'system-ui'],
    },
  },
  plugins: [forms],
} 