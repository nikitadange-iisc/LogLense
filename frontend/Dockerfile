# syntax=docker/dockerfile:1
#
# LogSense frontend (Vite + React) built into static assets and served by nginx.
# nginx also reverse-proxies /api to the backend container, replacing the Vite
# dev-server proxy used in local development.
#

# ---- Build stage -----------------------------------------------------------
FROM node:20-alpine AS build
WORKDIR /app

# Install dependencies from a clean lockfile (node_modules is .dockerignored,
# so this is always a fresh, platform-correct install — no committed binaries).
COPY package.json package-lock.json ./
RUN npm ci

# Build the production bundle.
COPY . .
RUN npm run build

# ---- Serve stage -----------------------------------------------------------
FROM nginx:alpine AS serve

# Custom config: SPA fallback + /api reverse proxy to the backend service.
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Static assets produced by `vite build`.
COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
