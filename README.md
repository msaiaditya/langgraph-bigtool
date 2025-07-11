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
- 🚀 **No Native Dependencies** - MemoryVectorStore works in all environments
- 🌐 **HTTP Embeddings** - Support for external embedding services
- ⚡ **Default Tools** - Specify tools that are always available without retrieval

## Installation

```bash
npm install langgraph-bigtool
```

## Quick Start

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

#### Option 1: MemoryVectorStore (Recommended - No native dependencies)
```typescript
import { MemoryVectorStore, HTTPEmbeddings } from "langgraph-bigtool";

// Create embeddings and store
const embeddings = new HTTPEmbeddings({ serviceUrl: 'http://localhost:8001' });
const store = new MemoryVectorStore(embeddings);

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

#### Option 2: HNSWLibStore (High performance - Requires Python and build tools)
```typescript
import { HNSWLibStore, HTTPEmbeddings } from "langgraph-bigtool";

// Create embeddings and store  
const embeddings = new HTTPEmbeddings({ serviceUrl: 'http://localhost:8001' });
const store = new HNSWLibStore(embeddings);

// Tools are indexed automatically when creating the agent
const agent = await createAgent({
  llm,
  tools: toolRegistry,
  store
});
```

**Benefits:**
- High-performance HNSW algorithm
- Better for very large tool sets (10,000+)
- Faster search times for complex queries

**Requirements:**
- Python 3.x installed
- C++ build tools
- May have issues on Alpine Linux or serverless environments

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
- The store parameter is optional. If you want semantic search for tools, provide a store instance (e.g., InMemoryStore or HNSWLibStore).
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
import { createAgent, MemoryVectorStore, HTTPEmbeddings } from "langgraph-bigtool";
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
const embeddings = new HTTPEmbeddings();
const store = new MemoryVectorStore(embeddings);

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

#### MemoryVectorStore (Recommended)

In-memory vector store that works in all environments without native dependencies:

```typescript
import { createAgent, MemoryVectorStore, HTTPEmbeddings } from "langgraph-bigtool";
import { OpenAIEmbeddings } from "@langchain/openai";

// Works with any embeddings provider
const embeddings = new OpenAIEmbeddings({ model: "text-embedding-3-small" });
// Or use HTTP embeddings
const embeddings = new HTTPEmbeddings({ serviceUrl: 'http://localhost:8001' });

// Create store - no configuration needed
const store = new MemoryVectorStore(embeddings);

// Create agent
const agent = await createAgent({
  llm,
  tools: toolRegistry,
  store
});
```

#### HNSWLibStore (High Performance)

For high-performance vector search with HNSW algorithm. **Note: Requires Python and build tools.**

```typescript
import { createAgent, HNSWLibStore, HTTPEmbeddings } from "langgraph-bigtool";

// Create embeddings
const embeddings = new HTTPEmbeddings({ serviceUrl: 'http://localhost:8001' });

// Create store with HNSW configuration
const store = new HNSWLibStore(embeddings, {
  space: 'cosine',      // 'cosine' | 'l2' | 'ip', defaults to 'cosine'
  numDimensions: 384    // Optional, auto-detected from first embedding
});

// Create agent
const agent = await createAgent({
  llm,
  tools: toolRegistry,
  store
});
```

### HTTPEmbeddings

HTTP-based embeddings client for environments where native dependencies are problematic (e.g., Alpine Linux, serverless environments):

```typescript
import { HTTPEmbeddings } from "langgraph-bigtool";

const embeddings = new HTTPEmbeddings({ 
  serviceUrl: 'http://localhost:8001' // Default value, can use env var EMBEDDINGS_SERVICE_URL
});

// Use with any vector store
const store = new MemoryVectorStore(embeddings);
```

**Required Endpoints:**
- `POST /embed` - Accepts `{ texts: string[] }` and returns `{ embeddings: number[][] }`
- `GET /health` - Returns 200 OK when service is healthy

The embeddings service should return vectors compatible with your model (e.g., 384 dimensions for all-MiniLM-L6-v2).

## Testing

### Running E2E Tests with MemoryVectorStore

To test the MemoryVectorStore with HTTP embeddings:

1. Start your embeddings service at `http://localhost:8001`
2. Run the E2E test:
   ```bash
   ./run-e2e-test.sh
   ```

The test will verify:
- Document indexing and storage
- Semantic search functionality
- Similarity scoring
- Integration with HTTP embeddings

## Examples

See the `examples/` directory for complete examples:
- `math.ts` - Mathematical operations with dynamic tool selection
- `default-tools.ts` - Using default tools alongside retrievable tools
- `custom-retrieval-store.ts` - Custom retrieval with keyword and embeddings search
- `hnswlib-store.ts` - Using HNSWLibStore for high-performance vector search

## License

MIT