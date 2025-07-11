import { HTTPEmbeddings } from "../src/embeddings/http.js";
import { MemoryVectorStore } from "../src/stores/MemoryVectorStore.js";
import { ToolRegistry } from "../src/types.js";
import { DynamicStructuredTool } from "@langchain/core/tools";
import { z } from "zod";

describe("MemoryVectorStore E2E Tests", () => {
  let embeddings: HTTPEmbeddings;
  let store: MemoryVectorStore;
  let toolRegistry: ToolRegistry;

  beforeAll(async () => {
    // Initialize HTTP embeddings with localhost service and verbose mode
    embeddings = new HTTPEmbeddings({ 
      serviceUrl: "http://localhost:8001",
      verbose: true 
    });
    
    // Check if embeddings service is running
    const isHealthy = await embeddings.checkHealth();
    if (!isHealthy) {
      throw new Error(
        "Embeddings service is not running. Please start the service at http://localhost:8001"
      );
    }
    console.log("✓ Embeddings service is healthy");
  });

  beforeEach(() => {
    // Create a fresh store for each test
    store = new MemoryVectorStore(embeddings);
    
    // Create test tool registry with diverse tools
    toolRegistry = {
      // Math tools
      add: new DynamicStructuredTool({
        name: "add",
        description: "Add two numbers together to get their sum",
        schema: z.object({
          a: z.number(),
          b: z.number(),
        }),
        func: async ({ a, b }) => `${a + b}`,
      }),
      subtract: new DynamicStructuredTool({
        name: "subtract",
        description: "Subtract one number from another to get the difference",
        schema: z.object({
          a: z.number(),
          b: z.number(),
        }),
        func: async ({ a, b }) => `${a - b}`,
      }),
      multiply: new DynamicStructuredTool({
        name: "multiply",
        description: "Multiply two numbers to get their product",
        schema: z.object({
          a: z.number(),
          b: z.number(),
        }),
        func: async ({ a, b }) => `${a * b}`,
      }),
      sqrt: new DynamicStructuredTool({
        name: "sqrt",
        description: "Calculate the square root of a number",
        schema: z.object({
          n: z.number(),
        }),
        func: async ({ n }) => `${Math.sqrt(n)}`,
      }),
      
      // String tools
      concat: new DynamicStructuredTool({
        name: "concat",
        description: "Concatenate two strings together",
        schema: z.object({
          a: z.string(),
          b: z.string(),
        }),
        func: async ({ a, b }) => a + b,
      }),
      uppercase: new DynamicStructuredTool({
        name: "uppercase",
        description: "Convert a string to uppercase letters",
        schema: z.object({
          text: z.string(),
        }),
        func: async ({ text }) => text.toUpperCase(),
      }),
      
      // Array tools
      sort: new DynamicStructuredTool({
        name: "sort",
        description: "Sort an array of numbers in ascending order",
        schema: z.object({
          numbers: z.array(z.number()),
        }),
        func: async ({ numbers }) => JSON.stringify(numbers.sort((a, b) => a - b)),
      }),
      filter: new DynamicStructuredTool({
        name: "filter",
        description: "Filter an array to keep only positive numbers",
        schema: z.object({
          numbers: z.array(z.number()),
        }),
        func: async ({ numbers }) => JSON.stringify(numbers.filter(n => n > 0)),
      }),
    };
  });

  test("should index tools successfully", async () => {
    console.log("\n=== Testing tool indexing ===");
    
    await store.indexTools(toolRegistry);
    
    // Verify tools are stored
    const allTools = await store.list(["tools"]);
    console.log(`Indexed ${allTools.length} tools`);
    expect(allTools.length).toBe(Object.keys(toolRegistry).length);
  });

  test("should generate embeddings for queries", async () => {
    console.log("\n=== Testing embedding generation ===");
    
    const testQuery = "calculate the sum of numbers";
    const embedding = await embeddings.embedQuery(testQuery);
    
    console.log(`Generated embedding for "${testQuery}"`);
    console.log(`Embedding dimensions: ${embedding.length}`);
    console.log(`First 5 values: ${embedding.slice(0, 5).join(", ")}...`);
    
    expect(embedding).toBeDefined();
    expect(Array.isArray(embedding)).toBe(true);
    expect(embedding.length).toBeGreaterThan(0);
    expect(embedding.every(n => typeof n === "number")).toBe(true);
  });

  test("should search for tools by exact name", async () => {
    console.log("\n=== Testing exact name search ===");
    
    await store.indexTools(toolRegistry);
    
    const results = await store.search(["tools"], {
      query: "add",
      limit: 3,
    });
    
    console.log(`Search results for "add":`);
    results.forEach((result, i) => {
      console.log(`  ${i + 1}. ${result.value.name} (score: ${result.score})`);
    });
    
    expect(results.length).toBeGreaterThan(0);
    expect(results[0].value.name).toBe("add");
    expect(results[0].score).toBeGreaterThan(0);
  });

  test("should search for tools by semantic meaning", async () => {
    console.log("\n=== Testing semantic search ===");
    
    await store.indexTools(toolRegistry);
    
    const queries = [
      { query: "calculate sum", expected: ["add"] },
      { query: "mathematical operations", expected: ["add", "subtract", "multiply"] },
      { query: "text manipulation", expected: ["concat", "uppercase"] },
      { query: "square root calculation", expected: ["sqrt"] },
      { query: "array operations", expected: ["sort", "filter"] },
    ];
    
    for (const { query, expected } of queries) {
      console.log(`\nSearching for: "${query}"`);
      
      const results = await store.search(["tools"], {
        query,
        limit: 3,
      });
      
      console.log(`Results:`);
      results.forEach((result, i) => {
        console.log(`  ${i + 1}. ${result.value.name} (score: ${result.score.toFixed(4)})`);
      });
      
      expect(results.length).toBeGreaterThan(0);
      
      // Check if at least one expected tool is in top results
      const resultNames = results.map(r => r.value.name);
      const hasExpectedTool = expected.some(toolName => 
        resultNames.includes(toolName)
      );
      expect(hasExpectedTool).toBe(true);
    }
  });

  test("should handle empty query", async () => {
    console.log("\n=== Testing empty query ===");
    
    await store.indexTools(toolRegistry);
    
    const results = await store.search(["tools"], {
      query: "",
      limit: 5,
    });
    
    console.log(`Results for empty query: ${results.length} tools returned`);
    expect(results.length).toBe(5);
    expect(results.every(r => r.score === 1)).toBe(true);
  });

  test("should respect limit parameter", async () => {
    console.log("\n=== Testing limit parameter ===");
    
    await store.indexTools(toolRegistry);
    
    const limits = [1, 3, 5, 10];
    
    for (const limit of limits) {
      const results = await store.search(["tools"], {
        query: "operations",
        limit,
      });
      
      console.log(`Limit ${limit}: returned ${results.length} results`);
      expect(results.length).toBeLessThanOrEqual(limit);
    }
  });

  test("should return meaningful similarity scores", async () => {
    console.log("\n=== Testing similarity scores ===");
    
    await store.indexTools(toolRegistry);
    
    const results = await store.search(["tools"], {
      query: "add two numbers together",
      limit: 5,
    });
    
    console.log("\nSimilarity scores:");
    results.forEach((result, i) => {
      console.log(`  ${i + 1}. ${result.value.name}: ${result.score.toFixed(6)}`);
      console.log(`     Description: ${result.value.description}`);
    });
    
    // Scores should be between 0 and 1 (or sometimes slightly above 1 due to cosine similarity)
    expect(results.every(r => r.score >= 0 && r.score <= 2)).toBe(true);
    
    // The most relevant tool should have the highest score
    expect(results[0].value.name).toBe("add");
    
    // Scores should be in descending order
    for (let i = 1; i < results.length; i++) {
      expect(results[i].score).toBeLessThanOrEqual(results[i - 1].score);
    }
  });

  test("should debug vector search process", async () => {
    console.log("\n=== Debugging vector search ===");
    
    await store.indexTools(toolRegistry);
    
    const query = "calculate the sum";
    console.log(`Query: "${query}"`);
    
    // Generate query embedding
    const queryEmbedding = await embeddings.embedQuery(query);
    console.log(`Query embedding dimensions: ${queryEmbedding.length}`);
    console.log(`Query embedding sample: [${queryEmbedding.slice(0, 3).join(", ")}...]`);
    
    // Perform search with detailed logging
    const results = await store.search(["tools"], {
      query,
      limit: 3,
    });
    
    console.log("\nSearch results:");
    results.forEach((result, i) => {
      console.log(`  ${i + 1}. Tool: ${result.value.name}`);
      console.log(`     Score: ${result.score}`);
      console.log(`     Description: ${result.value.description}`);
    });
    
    expect(results.length).toBeGreaterThan(0);
  });
});