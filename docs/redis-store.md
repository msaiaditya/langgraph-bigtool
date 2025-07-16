# Redis Store with Vector Search

The `RedisStore` provides persistent vector storage for semantic tool search using Redis and Redis Vector Search capabilities.

## Features

- **Persistent Storage**: Tools and their embeddings are stored in Redis
- **Intelligent Caching**: Only generates embeddings for new or changed tools
- **Batch Operations**: Efficient bulk checking and indexing
- **TTL Support**: Automatic expiration of unused tools (default: 7 days)
- **Multiple Embedding Providers**: Works with OpenAI, HTTP, or any LangChain embeddings

## Installation

```bash
npm install @langchain/redis redis
```

## Quick Start

### 1. Start Redis

```bash
docker-compose up -d redis
```

### 2. Basic Usage with OpenAI Embeddings

```typescript
import { RedisStore } from "langgraph-bigtool";
import { OpenAIEmbeddings } from "@langchain/openai";

const embeddings = new OpenAIEmbeddings({
  model: "text-embedding-3-small"
});

const store = new RedisStore({
  redisUrl: "redis://localhost:6379",
  embeddings,
  indexName: "my-tools",
  ttlSeconds: 7 * 24 * 60 * 60, // 7 days
  verbose: true
});

await store.connect();

// Use with createAgent
const agent = await createAgent({
  llm,
  tools,
  store
});
```

### 3. Using with HTTP Embeddings (No API Key Required)

```typescript
import { RedisStore, HTTPEmbeddings } from "langgraph-bigtool";

// Start embeddings service first:
// cd embeddings-service && docker-compose up -d

const embeddings = new HTTPEmbeddings({
  serviceUrl: 'http://localhost:8001'
});

const store = new RedisStore({
  redisUrl: "redis://localhost:6379",
  embeddings,
  indexName: "my-tools"
});
```

## Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `redisUrl` | string | required | Redis connection URL |
| `embeddings` | Embeddings | required | Any LangChain embeddings implementation |
| `indexName` | string | "bigtool-tools" | Name for the Redis vector index |
| `ttlSeconds` | number | 604800 (7 days) | Time-to-live for tool metadata |
| `verbose` | boolean | false | Enable performance logging |

## How It Works

### Intelligent Caching

The RedisStore uses SHA256 hashing to detect changes:

1. When indexing tools, it creates a hash of each tool's name and description
2. Before generating embeddings, it checks if the tool already exists with the same hash
3. Only new or changed tools have embeddings generated
4. This significantly reduces costs and improves performance

### Storage Structure

```
# Tool metadata (with TTL)
Key: bigtool:tools:meta:{tool_id}
Value: {
  tool_id: string,
  name: string,
  description: string,
  content_hash: string,
  indexed_at: number
}

# Vector embeddings (via RedisVectorStore)
Stored in Redis vector index for similarity search
```

### Performance

From production testing:
- Initial indexing: ~380ms per tool (including embedding generation)
- Cached indexing: ~1.3ms per tool (hash check only)
- Semantic search: 10-50ms depending on index size
- Batch operations: Process 20+ tools in single Redis round-trip

## Advanced Usage

### Get Index Statistics

```typescript
const stats = await store.getIndexStats();
console.log(`Total tools: ${stats.total}`);
console.log(`Oldest: ${stats.oldestIndexed}`);
console.log(`Newest: ${stats.newestIndexed}`);
```

### Clear All Tools

```typescript
await store.clearTools();
```

### Direct Semantic Search

```typescript
const results = await store.search(["tools"], {
  query: "text manipulation",
  limit: 5
});

results.forEach(result => {
  console.log(`${result.value.name}: ${result.score}`);
});
```

## Testing

Run the Redis store tests:

```bash
# Start Redis first
docker-compose up -d redis

# Run all Redis tests
npm run test:redis:all

# Run specific test suites
npm run test:redis        # OpenAI embeddings tests
npm run test:redis:http   # HTTP embeddings tests
```

## Examples

See complete examples in the `examples/` directory:

- `redis-openai.ts` - Full example with OpenAI embeddings
- `redis-http-embeddings.ts` - Using HTTP embeddings service
- `redis-http-quick-start.ts` - Minimal setup example

## Troubleshooting

### Redis Connection Issues

```
Error: Redis connection failed
```

Solution: Ensure Redis is running:
```bash
docker-compose up -d redis
docker ps | grep redis
```

### Embeddings Service Not Available

```
Error: Embeddings service is not available
```

Solution: Start the embeddings service:
```bash
cd embeddings-service
docker-compose up -d
```

### Type Mismatch with RedisVectorStore

The Redis client type from `redis` package may not match what `@langchain/redis` expects. This is handled internally with type casting.

## Production Considerations

1. **Redis Persistence**: Configure Redis with appropriate persistence settings
2. **TTL Strategy**: Adjust TTL based on your tool update frequency
3. **Index Optimization**: Create appropriate Redis indices for your query patterns
4. **Monitoring**: Use `verbose: true` in development, disable in production
5. **Scaling**: Redis Cluster is supported for horizontal scaling