# TA Frontend

Expo Router frontend for a medical insurance workflow with separate user and admin shells.

## Structure

- `app/`: routing only. Route files should stay thin and delegate real UI to `features/`.
- `features/`: screen and feature ownership.
- `contexts/`: app-wide state providers such as auth.
- `services/api/`: API code split by domain (`auth`, `claims`, `ai`) plus shared `client`, `config`, `storage`, and `types`.
- `components/`: reusable UI primitives that are still shared across features.

## Routing Model

- Public auth routes live at `/login` and `/register`.
- Protected user routes live under `/user/...`.
- Protected admin routes live under `/admin/...`.
- `app/index.tsx` is the auth gate that routes users into the correct shell.
- `app/user/_layout.tsx` and `app/admin/_layout.tsx` enforce role access so users cannot drift into the wrong namespace.

## Commands

```bash
npm install
npm run lint
npx tsc --noEmit
npx expo start
```

## Notes

- `app.config.js` wraps `app.json` to inject LAN-friendly local API values during development.
- Generated local export output such as `.expo-export-android/` should stay uncommitted.
