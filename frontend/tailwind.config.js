/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './src/pages/**/*.{js,ts,jsx,tsx}',
    './src/components/**/*.{js,ts,jsx,tsx}',
  ],
  theme: {
    extend: {
      colors: {
        card: 'white',
        'card-foreground': '#111827',
        'muted-foreground': '#6b7280',
      },
    },
  },
  plugins: [],
}
