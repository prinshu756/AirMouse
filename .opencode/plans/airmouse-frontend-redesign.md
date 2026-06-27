# AirMouse Frontend Redesign Plan

## Architecture
```
Vercel (edge) → Static Next.js site
  ↓
Browser (client) → MediaPipe.js WASM → getUserMedia → Canvas
  ↓
WIMouse.py (unchanged) → desktop mouse control via PyAutoGUI
```

## Files to create/modify

### 1. `airmouse-web/app/page.tsx` — Complete rewrite
- Soothing color scheme (indigo/slate bg, teal accents, glassmorphism)
- 4 screens: `start`, `loading`, `live`, `error`
- Start screen: glass card with Start Camera button + gesture preview
- Loading screen: spinner with status messages
- Live screen: camera feed + hand landmarks + stats panel + gesture guide
- Error screen: friendly error with retry for permission denied / no camera
- Beautiful hand landmark rendering: per-finger pastel colors, glow effects, sized by joint type
- Cursor trail: trailing glow circles that fade out
- Arrow functions, modern React patterns

### 2. `airmouse-web/app/globals.css` — Complete rewrite
- Ambient background with 3 floating gradient orbs (blur-heavy)
- Glassmorphism cards: `rgba(255,255,255,0.05)` bg + `backdrop-filter: blur(20px)`
- Soothing color palette:
  - Bg: `#0a0a14` → `#12121e` gradient
  - Glass: `rgba(255,255,255,0.04)` with `rgba(255,255,255,0.08)` borders
  - Accent: `#5eead4` (teal), `#22d3ee` (cyan), `#a78bfa` (purple)
  - Text: `#e2e8f0` primary, `#94a3b8` secondary
- Animations: fadeIn, pulse for live dot, spinner rotation, glow pulse on hover
- Responsive: mobile-first, reflows for tablets and desktops

### 3. `airmouse-web/public/favicon.svg` — New file
- Simple hand icon SVG for browser tab

### 4. `airmouse-web/next.config.mjs` — Update for static export
- Add `output: 'export'` for pure static hosting on Vercel

### 5. `airmouse-web/app/error.tsx` — New error boundary
- Catches runtime errors, shows friendly fallback

### 6. `airmouse-web/README.md` — Update with Vercel deploy button

## Implementation Order
1. page.tsx (main component with all screens + hand tracking + trail)
2. globals.css (all styles, animations, responsive breakpoints)
3. favicon.svg (simple icon)
4. next.config.mjs (static export)
5. error.tsx (error boundary)
6. README.md (deploy docs)
7. npm run build → verify zero errors/warnings
8. npm run dev → visual review and iteration

## Verification
- `npm run build` must succeed with zero errors
- Test in Chrome, Firefox, Edge
- Camera permission prompt must appear on button click
- Hand landmarks must render smoothly
- Error screen must show on permission denial
- Vercel deploy: `npx vercel --prod` from airmouse-web/
