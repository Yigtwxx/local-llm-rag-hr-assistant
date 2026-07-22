# Build context is the repository root — see .dockerignore.
#
# Base images are pinned to exact patch versions so a rebuild months from now
# produces the same image; bumping them is a deliberate edit, not a side effect
# of `docker compose build`.
FROM node:22.23.1-alpine AS build

WORKDIR /app

# Lockfile-only layer: cached until dependencies change.
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

COPY frontend/ ./
# `npm run build` runs `tsc -b` first, so a type error fails the image build.
RUN npm run build

FROM nginx:1.31.3-alpine

COPY frontend/nginx.conf /etc/nginx/conf.d/default.conf
COPY frontend/security-headers.conf /etc/nginx/snippets/security-headers.conf
COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 80
