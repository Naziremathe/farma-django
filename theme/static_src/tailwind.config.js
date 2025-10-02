module.exports = {
  content: [
    '../templates/**/*.html',
    '../../templates/**/*.html',
    '../../**/templates/**/*.html',
  ],
  theme: {
    extend: {
      colors: {
        primary: '#22c55e',
        'primary-dark': '#16a34a',
      },
    },
  },
  plugins: [
    require('daisyui'),
  ],
}
