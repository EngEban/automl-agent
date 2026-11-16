# AutoML Agent frontend foundation

The frontend is a Next.js App Router application using React, TypeScript, Tailwind CSS, and Lucide React. This task provides the shared visual foundation and a development-only design-system preview; product pages and authentication flows are not implemented here.

## Requirements

- Node.js 24 or later.
- npm 11.12.1. Exact direct dependency versions are pinned in package.json; keep the generated package-lock.json with the manifest after installation.

Install dependencies and start the development server from this directory:

    npm install
    npm run dev

Open http://localhost:3000/design-preview to inspect the design system. The route returns not found outside development. The root route links to the preview only in development.

Copy .env.example to .env.local only when you want to configure the API client. NEXT_PUBLIC_API_BASE_URL is the backend origin (default http://localhost:8100); the API client adds the existing /api prefix. It is not a secret. The client accepts a bearer token per request because the backend returns Authorization bearer tokens and does not set an authentication cookie. It does not persist tokens.

For the current API contracts, use docs/API.md and the Pydantic schemas under backend/app/schemas. Cross-origin browser requests also require the backend CORS_ORIGINS setting to allow http://localhost:3000.

## Checks

    npm run typecheck
    npm run lint
    npm run build

There is no frontend test framework in this repository yet, so no component test dependency was added.
