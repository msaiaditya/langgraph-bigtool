#!/bin/bash

echo "🧪 Running MemoryVectorStore E2E Tests"
echo "======================================"
echo ""
echo "Prerequisites:"
echo "- HTTP embeddings service must be running at http://localhost:8001"
echo "- Service should expose /embed and /health endpoints"
echo ""

# Check if embeddings service is running
echo "Checking embeddings service health..."
curl -s -o /dev/null -w "%{http_code}" http://localhost:8001/health | grep -q "200"
if [ $? -ne 0 ]; then
    echo "❌ Embeddings service is not running at http://localhost:8001"
    echo ""
    echo "Please start your embeddings service with:"
    echo "- POST /embed endpoint that accepts { texts: string[] }"
    echo "- GET /health endpoint that returns 200 OK"
    exit 1
fi

echo "✅ Embeddings service is healthy"
echo ""

# Build the project first
echo "Building project..."
npm run build

# Run the specific E2E test
echo "Running E2E tests..."
npx jest tests/memory-vector-store-e2e.test.ts --verbose --no-coverage

echo ""
echo "Test complete!"