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

// Create tool registry (option 1: manual)
const toolRegistry = {
  get_weather: weatherTool,
  // ... add more tools
};

// Or use helper function (option 2: from array)
import { createToolRegistry } from "langgraph-bigtool";
const tools = [weatherTool, calculatorTool];
const toolRegistry = createToolRegistry(tools);

// Create agent
const llm = new ChatOpenAI({ model: "gpt-4" });
const agent = await createAgent({
  llm,
  tools: toolRegistry,
  options: { limit: 2 }
});

// Use the agent
const result = await agent.invoke({
  messages: [{ role: "user", content: "What's the weather in London?" }],
  selected_tool_ids: []
});
```

## How It Works

1. **Tool Registry**: All available tools are stored in a registry (not loaded into the agent initially)
2. **Tool Retrieval**: The agent has a special `retrieve_tools` tool that can search for relevant tools
3. **Dynamic Loading**: When the agent needs tools, it searches for them and only loads the relevant ones
4. **Execution**: The agent can then use the retrieved tools to complete the task

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
  - `store?`: Optional BaseStore instance for tool search
  - `prompt?`: Optional system prompt
  - `options?`: Optional configuration
    - `limit`: Maximum number of tools to retrieve per search (default: 2)
    - `filter`: Optional filter for tool searches
    - `namespace_prefix`: Store namespace for tools (default: ["tools"])
    - `retrieve_tools_function`: Custom function for tool retrieval

**Returns:** A compiled LangGraph agent ready to use

**Note:** The store parameter is optional. If you want semantic search for tools, provide a store instance (e.g., InMemoryStore or HNSWLibStore).

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
- `custom-retrieval-store.ts` - Custom retrieval with keyword and embeddings search
- `hnswlib-store.ts` - Using HNSWLibStore for high-performance vector search

## License

MIT