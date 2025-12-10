/** @type {import('tailwindcss').Config} */
export default {
    content: [
        "./index.html",
        "./src/**/*.{js,ts,jsx,tsx}",
    ],
    theme: {
        extend: {
            colors: {
                'primary': '#121519',    // Background
                'accent': '#448c74',     // Teal Accent
                'accent-dim': 'rgba(68, 140, 116, 0.1)',
                'accent-glow': 'rgba(68, 140, 116, 0.5)',
                // Custom white/transparent are not needed in extend as they exist in default, 
                // but we can add aliases if we want.
            },
            boxShadow: {
                'neon': '0 0 10px rgba(68, 140, 116, 0.5), 0 0 20px rgba(68, 140, 116, 0.3)',
                'neon-strong': '0 0 15px rgba(68, 140, 116, 0.8), 0 0 30px rgba(68, 140, 116, 0.4)',
                'glass': '0 8px 32px 0 rgba(0, 0, 0, 0.5)',
            },
            fontFamily: {
                'mono': ['JetBrains Mono', 'Fira Code', 'monospace'],
                'sans': ['Inter', 'system-ui', 'sans-serif'],
            }
        },
    },
    plugins: [],
}
