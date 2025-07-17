# langgraph-bigtool

Build LangGraph agents with large numbers of tools using dynamic tool selection.

## Overview

`langgraph-bigtool` is a TypeScript library that enables LangGraph agents to dynamically discover and use tools from large registries through semantic search. Instead of loading all tools into the agent's context at once, this library allows agents to search for and retrieve only the relevant tools they need for a given task.

## Features

- 🔍 **Dynamic Tool Discovery** - Agents can search for tools based on natural language queries
- 📈 **Scalable** - Handle hundreds or thousands of tools efficiently
- 🎯 **Semantic Search** - Use vector stores to find the most relevant tools
- 🔧 **Flexible Retrieval** - Customize how tools are discovered and retrieved
- 📦 **LangGraph Compatible** - Built on top of LangGraph's proven architecture
- 🚀 **No Native Dependencies** - MemoryVectorBaseStore works in all environments
- 🗄️ **Redis Support** - Production-ready persistent storage with intelligent caching
- ⚡ **Redis Embedding Cache** - Cache computed embeddings in Redis for 10-20x faster startup
- 🌐 **HTTP Embeddings** - Support for external embedding services
- ⚡ **Default Tools** - Specify tools that are always available without retrieval

## Installation

```bash
npm install langgraph-bigtool
```

## Quick Start

> **Note**: For production use, we recommend using RedisVectorBaseStore for persistent vector storage. See the [Vector Stores](#vector-stores) section for details.

```typescript
import { createAgent } from "langgraph-bigtool";
import { ChatOpenAI } from "@langchain/openai";
import { tool } from "@langchain/core/tools";
import { z } from "zod";

// Create your tools
const weatherTool = tool(
  async ({ city }) => `The weather in ${city} is sunny`,
  {
    name: "get_weather",
    description: "Get the weather for a city",
    schema: z.object({
      city: z.string()
    })
  }
);

const calculatorTool = tool(
  async ({ operation, a, b }) => {
    switch(operation) {
      case "add": return a + b;
      case "subtract": return a - b;
      case "multiply": return a * b;
      case "divide": return a / b;
    }
  },
  {
    name: "calculator",
    description: "Basic calculator operations",
    schema: z.object({
      operation: z.enum(["add", "subtract", "multiply", "divide"]),
      a: z.number(),
      b: z.number()
    })
  }
);

// Create tool registries
const toolRegistry = {
  get_weather: weatherTool,
  // ... more tools that need to be discovered
};

const defaultTools = {
  calculator: calculatorTool  // Always available without retrieval
};

// Create agent with default tools
const llm = new ChatOpenAI({ model: "gpt-4" });
const agent = await createAgent({
  llm,
  tools: toolRegistry,
  defaultTools,  // These tools are always available
  options: { limit: 2 }
});

// Use the agent - calculator is immediately available
const result = await agent.invoke({
  messages: [{ role: "user", content: "What's 15 + 27?" }],
  selected_tool_ids: []
});
```

## How It Works

1. **Tool Registry**: All available tools are stored in a registry (not loaded into the agent initially)
2. **Tool Retrieval**: The agent has a special `retrieve_tools` tool that can search for relevant tools
3. **Dynamic Loading**: When the agent needs tools, it searches for them and only loads the relevant ones
4. **Execution**: The agent can then use the retrieved tools to complete the task

## Default Tools

Default tools are tools that are always available to the agent without needing to be retrieved. This is useful for:

- **Common operations** that are frequently used (e.g., calculator, time/date functions)
- **Utility tools** that support other operations (e.g., formatters, validators)
- **Core functionality** that should always be accessible

### How Default Tools Work

```typescript
const defaultTools = {
  calculator: calculatorTool,
  getCurrentTime: timeTool
};

const toolRegistry = {
  weatherAPI: weatherTool,
  databaseQuery: dbTool,
  // ... hundreds more specialized tools
};

const agent = await createAgent({
  llm,
  tools: toolRegistry,      // Tools that need to be discovered
  defaultTools,             // Always available, not indexed
  store
});
```

Key differences:
- **Default tools** are always bound to the LLM alongside `retrieve_tools`
- **Registry tools** must be discovered via semantic search before use
- **Default tools** are not indexed in the vector store
- Both can be used together in the same conversation

### Using with a Store (Recommended for large tool sets)

To enable semantic search over your tools, provide a store:

#### Option 1: MemoryVectorBaseStore (Recommended - No native dependencies)
```typescript
import { MemoryVectorBaseStore } from "langgraph-bigtool";
import { OpenAIEmbeddings } from "@langchain/openai";

// Create embeddings and store
const embeddings = new OpenAIEmbeddings({
  model: "text-embedding-3-small",
  apiKey: process.env.OPENAI_API_KEY // or pass directly
});
const store = new MemoryVectorBaseStore(embeddings, {
  verbose: true // Optional: enable performance logging for vector operations
});

// Tools are indexed automatically when creating the agent
const agent = await createAgent({
  llm,
  tools: toolRegistry,
  store
});
```

**Benefits:**
- Works in all JavaScript environments (Node.js, browsers, serverless)
- No native dependencies or build tools required
- Suitable for up to thousands of tools
- Easy deployment and debugging

**Performance Monitoring:**
When `verbose: true` is set on MemoryVectorBaseStore, it logs:
- Tool indexing time: `[MemoryVectorBaseStore] Indexing tools completed in XXXms - N tools indexed`
- Search performance: `[MemoryVectorBaseStore] Vector similarity search completed in XXXms - found N results for query: "..."`

### Using without a Store

For small tool sets or when you don't need dynamic tool discovery:

```typescript
const agent = await createAgent({
  llm,
  tools: toolRegistry
  // No store - agent will have access to all tools immediately
});
```

## API Reference

### `createAgent(input)`

Creates a LangGraph agent with dynamic tool selection capabilities.

**Parameters:**
- `input`: Configuration object with the following properties:
  - `llm`: A LangChain chat model that supports tool calling
  - `tools`: Tool registry object or array of tools
  - `defaultTools?`: Optional tools that are always available without retrieval
  - `store?`: Optional BaseStore instance for tool search
  - `prompt?`: Optional system prompt
  - `options?`: Optional configuration
    - `limit`: Maximum number of tools to retrieve per search (default: 2)
    - `filter`: Optional filter for tool searches
    - `namespace_prefix`: Store namespace for tools (default: ["tools"])
    - `retrieve_tools_function`: Custom function for tool retrieval

**Returns:** A compiled LangGraph agent ready to use

**Notes:**
- The store parameter is optional. If you want semantic search for tools, provide a store instance (e.g., InMemoryStore or MemoryVectorBaseStore).
- Default tools are always available to the agent and are not indexed in the store.
- Both `tools` and `defaultTools` can accept either an object or an array of tools.

### Helper Functions

#### `createToolRegistry(tools)`

Convert an array of tools to a tool registry object.

```typescript
const tools = [tool1, tool2, tool3];
const registry = createToolRegistry(tools);
// Returns: { tool1_name: tool1, tool2_name: tool2, tool3_name: tool3 }
```

### Types

```typescript
// Tool registry type
type ToolRegistry = Record<string, StructuredTool>;

// Custom retrieval function
type RetrieveToolsFunction = (
  query: string,
  config?: {
    limit?: number;
    filter?: Record<string, any>;
  },
  store?: BaseStore
) => Promise<string[]>;

// Agent state
interface BigToolState {
  messages: BaseMessage[];
  selected_tool_ids: string[];
}
```

### Available Exports

```typescript
import {
  // Main function
  createAgent,
  
  // Stores
  MemoryVectorBaseStore,
  RedisVectorBaseStore,
  RedisCachedMemoryVectorBaseStore,
  
  // Embeddings
  // Note: Use OpenAIEmbeddings from @langchain/openai
  
  // Utilities
  createToolRegistry,
  
  // Types
  type ToolRegistry,
  type BigToolState,
  type RetrieveToolsFunction
} from "langgraph-bigtool";
```

## Advanced Usage

### Custom Tool Retrieval

You can provide your own tool retrieval logic:

```typescript
const customRetriever: RetrieveToolsFunction = async (query, config, store) => {
  // Your custom logic here
  // Return array of tool IDs
  return ["tool1", "tool2"];
};

const agent = await createAgent({
  llm,
  tools: toolRegistry,
  options: {
    retrieve_tools_function: customRetriever
  }
});
```

### Combining Default Tools with Dynamic Retrieval

Here's a complete example showing how to effectively use default tools alongside a large registry of specialized tools:

```typescript
import { createAgent, MemoryVectorBaseStore } from "langgraph-bigtool";
import { OpenAIEmbeddings } from "@langchain/openai";
import { tool } from "@langchain/core/tools";
import { z } from "zod";

// Default tools - always available, frequently used
const defaultTools = {
  // Basic calculator for common math operations
  calculator: tool(
    async ({ operation, a, b }) => {
      const ops = { add: a + b, subtract: a - b, multiply: a * b, divide: a / b };
      return `Result: ${ops[operation]}`;
    },
    {
      name: "calculator",
      description: "Basic calculator",
      schema: z.object({
        operation: z.enum(["add", "subtract", "multiply", "divide"]),
        a: z.number(),
        b: z.number()
      })
    }
  ),
  
  // Current time utility
  getCurrentTime: tool(
    async () => new Date().toISOString(),
    {
      name: "getCurrentTime",
      description: "Get current date and time",
      schema: z.object({})
    }
  )
};

// Specialized tools - discovered via semantic search
const specializedTools = {
  // Advanced math operations
  sqrt: createMathTool("sqrt", (x) => Math.sqrt(x)),
  factorial: createMathTool("factorial", factorial),
  fibonacci: createMathTool("fibonacci", fibonacci),
  
  // API tools
  weatherAPI: createAPITool("weather", "Get weather data"),
  stockAPI: createAPITool("stocks", "Get stock prices"),
  newsAPI: createAPITool("news", "Get latest news"),
  
  // Database tools
  queryUsers: createDBTool("users", "Query user database"),
  queryOrders: createDBTool("orders", "Query orders database"),
  
  // ... potentially hundreds more specialized tools
};

// Set up vector store for semantic search
const embeddings = new OpenAIEmbeddings({
  model: "text-embedding-3-small",
  apiKey: process.env.OPENAI_API_KEY // or pass directly
});
const store = new MemoryVectorBaseStore(embeddings);

// Create agent with both default and specialized tools
const agent = await createAgent({
  llm: new ChatOpenAI({ model: "gpt-4" }),
  tools: specializedTools,    // Large registry needing discovery
  defaultTools,               // Always available basics
  store,
  options: { 
    limit: 3  // Retrieve up to 3 specialized tools per search
  }
});

// Example usage
const result = await agent.invoke({
  messages: [
    { role: "user", content: "What's 25 + 17?" }, // Uses default calculator
    { role: "assistant", content: "25 + 17 = 42" },
    { role: "user", content: "Now find its square root" } // Needs to retrieve sqrt tool
  ],
  selected_tool_ids: []
});
```

**Best Practices for Default Tools:**
1. Keep default tools small and focused on common operations
2. Include tools that are used across many different tasks
3. Avoid making specialized or domain-specific tools default
4. Consider performance - default tools are loaded for every request


### Vector Stores

#### RedisVectorBaseStore (Recommended for Production)

Redis-based vector store with persistent storage, intelligent caching, and TTL support:

```typescript
import { createAgent, RedisVectorBaseStore } from "langgraph-bigtool";
import { OpenAIEmbeddings } from "@langchain/openai";

// Option 1: OpenAI embeddings
const openaiEmbeddings = new OpenAIEmbeddings({ 
  model: "text-embedding-3-small",
  apiKey: "your-api-key" // or use OPENAI_API_KEY env var
});

// Option 2: Local embeddings service (5-80x faster)
const localEmbeddings = new OpenAIEmbeddings({
  model: "text-embedding-3-small", // Model name (ignored by local service)
  apiKey: "not-needed", // Required by library but not used by local service
  configuration: {
    baseURL: 'http://localhost:8001/v1'
  }
});

// Create Redis store with intelligent caching
const store = new RedisVectorBaseStore({
  redisUrl: process.env.REDIS_URL || "redis://localhost:6379",
  embeddings: localEmbeddings,
  indexName: "bigtool-tools",
  ttlSeconds: 7 * 24 * 60 * 60, // 7 days
  verbose: true // Enable performance logging
});

// Connect to Redis
await store.connect();

// Create agent - tools are indexed automatically
const agent = await createAgent({
  llm,
  tools: toolRegistry,
  store
});

// Don't forget to disconnect when done
await store.disconnect();
```

**Benefits:**
- Persistent vector storage across restarts
- Intelligent caching avoids redundant embedding generation
- TTL (Time To Live) for automatic cleanup
- Batch operations for better performance
- RedisInsight web UI for monitoring
- Production-ready with connection pooling

**Setup Redis:**
```bash
# Using Docker Compose (recommended)
docker-compose up -d

# Or using Docker directly
docker run -d --name bigtool-redis -p 6379:6379 -p 8002:8001 redis/redis-stack:latest

# Access RedisInsight at http://localhost:8002
```

**Example with caching:**
```typescript
// First run: generates embeddings
await agent.invoke({
  messages: [{ role: "user", content: "Calculate the square root of 16" }]
});
// Output: [RedisVectorBaseStore] New tools to index: 8, Already cached: 0

// Second run: uses cached embeddings (much faster)
await agent.invoke({
  messages: [{ role: "user", content: "Find the cosine of 45 degrees" }]
});
// Output: [RedisVectorBaseStore] New tools to index: 0, Already cached: 8
```

See complete examples:
- [`examples/redis-openai.ts`](examples/redis-openai.ts) - Full Redis + OpenAI setup
- [`examples/redis-http-embeddings.ts`](examples/redis-http-embeddings.ts) - Redis + local embeddings service
- [`examples/redis-http-quick-start.ts`](examples/redis-http-quick-start.ts) - Minimal Redis setup
- [`examples/redis-embeddings-performance-comparison.ts`](examples/redis-embeddings-performance-comparison.ts) - Performance benchmarking

#### RedisCachedMemoryVectorBaseStore (Redis + In-Memory Hybrid)

Combines Redis caching with in-memory vector search for optimal performance. Embeddings are computed once, cached in Redis, and loaded into memory for fast searches:

```typescript
import { createAgent, RedisCachedMemoryVectorBaseStore } from "langgraph-bigtool";
import { OpenAIEmbeddings } from "@langchain/openai";
import { tool } from "@langchain/core/tools";
import { z } from "zod";

// Configure embeddings with environment variables
const embeddings = new OpenAIEmbeddings({
  model: process.env.EMBEDDINGS_MODEL || "text-embedding-3-small",
  apiKey: process.env.EMBEDDINGS_API_KEY || "not-needed",
  configuration: {
    baseURL: process.env.EMBEDDINGS_URL || "http://localhost:8001/v1"
  }
});

// Create Redis-cached memory store
const store = new RedisCachedMemoryVectorBaseStore({
  redisUrl: process.env.REDIS_URL || "redis://localhost:6379",
  embeddings,
  indexName: process.env.TOOL_INDEX_NAME || "my-app-tools",
  ttlSeconds: parseInt(process.env.CACHE_TTL_DAYS || "7") * 24 * 60 * 60,
  verbose: process.env.NODE_ENV !== "production"
});

// Connect to Redis
await store.connect();

// Create your tools
const tools = {
  calculateSum: tool(
    async ({ a, b }: { a: number; b: number }) => `${a} + ${b} = ${a + b}`,
    {
      name: "calculateSum",
      description: "Add two numbers together",
      schema: z.object({ a: z.number(), b: z.number() })
    }
  ),
  // ... more tools
};

// Create agent - tools are indexed automatically
const agent = await createAgent({
  llm: yourLLM,
  tools,
  store,
  maxToolRoundtrips: 2
});

// Use the agent
const result = await agent.invoke({
  messages: [{ role: "user", content: "What is 25 plus 17?" }]
});

// Clean up when done
await store.disconnect();
```

**Benefits:**
- **First run**: Computes embeddings and caches them in Redis
- **Subsequent runs**: Loads embeddings from cache (10-20x faster startup)
- **In-memory search**: All searches performed in memory for maximum speed
- **Persistent cache**: Embeddings survive application restarts
- **Batch operations**: Uses Redis MGET and pipelines for efficiency

**When to use:**
- Applications with many tools (10+) that restart frequently
- Development environments where you're iterating on code
- Serverless functions that need fast cold starts
- When you want to reduce embedding API costs

**Performance example:**
```
First run:  [RedisCachedMemoryVectorBaseStore] Indexed 50 tools in 523ms (0 cache hits, 50 computed)
Second run: [RedisCachedMemoryVectorBaseStore] Indexed 50 tools in 23ms (50 cache hits [100%], 0 computed)
```

#### MemoryVectorBaseStore (Development/Testing)

In-memory vector store that works in all environments without native dependencies:

```typescript
import { createAgent, MemoryVectorBaseStore } from "langgraph-bigtool";
import { OpenAIEmbeddings } from "@langchain/openai";

const embeddings = new OpenAIEmbeddings({
  model: "text-embedding-3-small", // Model name (ignored by local service)
  apiKey: "not-needed", // Required by library but not used by local service
  configuration: {
    baseURL: 'http://localhost:8001/v1'
  }
});

// Create store with optional performance logging
const store = new MemoryVectorBaseStore(embeddings, {
  verbose: true // See indexing and search performance
});

// Create agent
const agent = await createAgent({
  llm,
  tools: toolRegistry,
  store
});
```

### Using Local Embeddings Service

For environments where native dependencies are problematic or when you want to use a local embeddings service, you can configure OpenAIEmbeddings to point to your local OpenAI-compatible service:

```typescript
import { MemoryVectorBaseStore } from "langgraph-bigtool";
import { OpenAIEmbeddings } from "@langchain/openai";

const embeddings = new OpenAIEmbeddings({
  model: "text-embedding-3-small",  // Model name (ignored by local service)
  apiKey: "not-needed",  // Required by the library but not used by local service
  configuration: {
    baseURL: 'http://localhost:8001/v1'  // Point to your local embeddings service
  }
});

// Use with any vector store
const store = new MemoryVectorBaseStore(embeddings);
```

**About Local Embeddings Service:**
When using a local OpenAI-compatible embeddings service, you can configure `OpenAIEmbeddings` to point to your local endpoint by setting the `baseURL` in the configuration. The model name is typically ignored by local services that only serve one model.

**Local Service Requirements:**
Your local embeddings service should implement the OpenAI embeddings API format:
- `POST /v1/embeddings` - Accepts OpenAI-format requests and returns compatible responses
- The service should return vectors compatible with your model (e.g., 384 dimensions for all-MiniLM-L6-v2)

## Testing

### Running E2E Tests with MemoryVectorBaseStore

To test the MemoryVectorBaseStore with a local embeddings service:

1. Start your embeddings service at `http://localhost:8001`
2. Run the E2E test:
   ```bash
   ./run-e2e-test.sh
   ```

The test will verify:
- Document indexing and storage
- Semantic search functionality
- Similarity scoring
- Integration with local embeddings service

## Examples

See the `examples/` directory for complete examples:
- `math.ts` - Mathematical operations with dynamic tool selection
- `default-tools.ts` - Using default tools alongside retrievable tools
- `custom-retrieval-store.ts` - Custom retrieval with keyword and embeddings search
- `redis-openai.ts` - RedisVectorBaseStore with OpenAI embeddings and intelligent caching
- `redis-http-embeddings.ts` - RedisVectorBaseStore with local embeddings service (no API key required)
- `redis-http-quick-start.ts` - Minimal Redis setup example
- `redis-embeddings-performance-comparison.ts` - Performance benchmarking of embeddings providers
- `redis-cached-memory.ts` - RedisCachedMemoryVectorBaseStore example with performance comparison

## License

MIT