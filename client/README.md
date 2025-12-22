# Citation Retrieval Client

Next.js frontend for the Citation Retrieval system. Provides an interactive web interface for finding citations.

## Features

- Interactive citation finder with example contexts
- Real-time API integration with the backend
- APA and MLA citation formatting
- Confidence scoring and reasoning display
- Fallback to demo data when API is unavailable
- Dark mode support
- Responsive design

## Prerequisites

- Node.js 20 or higher
- pnpm (preferred) or npm
- Citation Retrieval API running (see [../server/README.md](../server/README.md))

## Local Development Setup

### 1. Install dependencies

```bash
# Using pnpm (recommended)
pnpm install

# Or using npm
npm install
```

### 2. Configure environment variables

Create a `.env.local` file:

```bash
echo "NEXT_PUBLIC_API_URL=http://localhost:8000" > .env.local
```

Or manually create `.env.local`:

```env
# API Configuration
NEXT_PUBLIC_API_URL=http://localhost:8000
```

### 3. Run the development server

```bash
# Using pnpm
pnpm dev

# Or using npm
npm run dev
```

The application will be available at http://localhost:3000

### 4. Build for production

```bash
# Build the application
pnpm build

# Start production server
pnpm start
```

## Docker Setup

### Build the Docker image

```bash
docker build -t citation-retrieval-client \
  --build-arg NEXT_PUBLIC_API_URL=http://api:8000 \
  .
```

### Run the container

```bash
docker run -p 3000:3000 \
  -e NEXT_PUBLIC_API_URL=http://localhost:8000 \
  citation-retrieval-client
```

### Using Docker Compose

See the main [docker-compose.yml](../docker-compose.yml) in the root directory:

```bash
# From the root directory
docker-compose up -d client
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `NEXT_PUBLIC_API_URL` | Backend API URL | `http://localhost:8000` |

## Project Structure

```
client/
├── app/                        # Next.js app directory
│   ├── page.tsx               # Home page with citation finder
│   ├── layout.tsx             # Root layout with theme provider
│   └── globals.css            # Global styles
├── components/
│   ├── citation-finder.tsx    # Main citation finder component
│   ├── theme-provider.tsx     # Dark mode theme provider
│   └── ui/                    # shadcn/ui components
├── lib/
│   └── utils.ts               # Utility functions
├── public/                    # Static assets
├── styles/                    # Additional styles
├── .env.local                 # Local environment variables
├── Dockerfile                 # Docker configuration
├── next.config.mjs            # Next.js configuration
├── package.json               # Dependencies and scripts
└── tsconfig.json              # TypeScript configuration
```

## API Integration

The client communicates with the backend API via REST. The main endpoint used is:

### POST `/api/find-citation`

Request:
```json
{
  "context": "The architecture [CITATION] introduced attention mechanisms.",
  "k": 5,
  "use_llm_reranker": true
}
```

Response:
```json
{
  "results": [
    {
      "citation": {
        "title": "Attention Is All You Need",
        "authors": ["Vaswani, A.", "..."],
        "year": 2017,
        "source": "NeurIPS",
        "doi": "...",
        "abstract": "..."
      },
      "confidence": 99.5,
      "reasoning": "...",
      "score": 0.98,
      "formatted": {
        "apa": "Vaswani, A. et al. (2017). ...",
        "mla": "Vaswani, A., et al. ..."
      }
    }
  ],
  "query": "The architecture introduced attention mechanisms.",
  "expanded_queries": ["..."]
}
```

## Development

### Code Style

The project uses:
- TypeScript for type safety
- Tailwind CSS for styling
- shadcn/ui for UI components
- Next.js App Router

### Adding New Features

1. Components go in `components/`
2. UI components use shadcn/ui patterns
3. API calls are made directly in components (consider adding a dedicated API client for larger apps)

### Linting

```bash
pnpm lint
```

## Troubleshooting

### API Connection Issues

If the client can't connect to the API:

1. Check that the API is running:
   ```bash
   curl http://localhost:8000/health
   ```

2. Verify `NEXT_PUBLIC_API_URL` in `.env.local`

3. Check browser console for CORS errors

4. The app will fallback to demo data if the API is unavailable

### Build Errors

If you encounter build errors:

```bash
# Clear Next.js cache
rm -rf .next

# Reinstall dependencies
rm -rf node_modules
pnpm install

# Rebuild
pnpm build
```

### TypeScript Errors

The project has `ignoreBuildErrors: true` in `next.config.mjs` for development convenience. For production builds, you may want to fix TypeScript errors.

## License

[Add your license information here]
