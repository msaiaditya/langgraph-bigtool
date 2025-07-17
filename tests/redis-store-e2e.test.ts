import { describe, it, expect, beforeAll, afterAll, beforeEach, jest } from "@jest/globals";
import { RedisVectorBaseStore } from "../src/stores/RedisVectorBaseStore.js";
import { OpenAIEmbeddings } from "@langchain/openai";
import { createClient, RedisClientType } from "redis";
import { createAgent } from "../src/index.js";
import { ChatOpenAI } from "@langchain/openai";
import { tool } from "@langchain/core/tools";
import { z } from "zod";
import { HumanMessage } from "@langchain/core/messages";
import type { ToolRegistry } from "../src/types.js";

describe("RedisVectorBaseStore E2E Tests", () => {
  let store: RedisVectorBaseStore;
  let redisClient: RedisClientType;
  const testIndexName = "test-bigtool-tools";
  
  beforeAll(async () => {
    // Setup Redis client for cleanup
    redisClient = createClient({ url: "redis://localhost:6379" });
    await redisClient.connect();
    
    // Initialize store
    const embeddings = new OpenAIEmbeddings({
      apiKey: "not-needed",
      configuration: { baseURL: 'http://localhost:8001/v1' }
    });
    
    store = new RedisVectorBaseStore({
      redisUrl: "redis://localhost:6379",
      embeddings,
      indexName: testIndexName,
      ttlSeconds: 60, // Short TTL for testing
      verbose: false
    });
    
    await store.connect();
  }, 30000);
  
  afterAll(async () => {
    // Cleanup
    const keys = await redisClient.keys(`bigtool:*`);
    if (keys.length > 0) {
      await redisClient.del(...keys);
    }
    await store.disconnect();
    await redisClient.quit();
  });
  
  beforeEach(async () => {
    // Clear test data before each test
    await store.clearTools();
  });
  
  it("should index tools only once with same content", async () => {
    const tools: ToolRegistry = {
      calculator: tool(
        async ({ a, b }: { a: number; b: number }) => `${a} + ${b} = ${a + b}`,
        {
          name: "calculator",
          description: "Adds two numbers",
          schema: z.object({ a: z.number(), b: z.number() })
        }
      )
    };
    
    // First indexing
    const spy = jest.spyOn(console, "log");
    await store.indexTools(tools);
    expect(spy).toHaveBeenCalledWith("Indexed 1 new/updated tools, 0 unchanged");
    
    // Second indexing - should skip
    spy.mockClear();
    await store.indexTools(tools);
    expect(spy).toHaveBeenCalledWith("Indexed 0 new/updated tools, 1 unchanged");
    
    spy.mockRestore();
  }, 30000);
  
  it("should detect tool changes and reindex", async () => {
    let tools: ToolRegistry = {
      calculator: tool(
        async ({ a, b }: { a: number; b: number }) => `${a} + ${b} = ${a + b}`,
        {
          name: "calculator",
          description: "Adds two numbers",
          schema: z.object({ a: z.number(), b: z.number() })
        }
      )
    };
    
    await store.indexTools(tools);
    
    // Change tool description
    tools = {
      calculator: tool(
        async ({ a, b }: { a: number; b: number }) => `${a} + ${b} = ${a + b}`,
        {
          name: "calculator",
          description: "Adds two numbers together", // Changed
          schema: z.object({ a: z.number(), b: z.number() })
        }
      )
    };
    
    const spy = jest.spyOn(console, "log");
    await store.indexTools(tools);
    expect(spy).toHaveBeenCalledWith("Indexed 1 new/updated tools, 0 unchanged");
    spy.mockRestore();
  }, 30000);
  
  it("should perform semantic search correctly", async () => {
    const tools: ToolRegistry = {
      add: tool(
        async ({ a, b }: { a: number; b: number }) => a + b,
        {
          name: "add",
          description: "Add two numbers together",
          schema: z.object({ a: z.number(), b: z.number() })
        }
      ),
      multiply: tool(
        async ({ a, b }: { a: number; b: number }) => a * b,
        {
          name: "multiply", 
          description: "Multiply two numbers",
          schema: z.object({ a: z.number(), b: z.number() })
        }
      ),
      sqrt: tool(
        async ({ n }: { n: number }) => Math.sqrt(n),
        {
          name: "sqrt",
          description: "Calculate square root of a number",
          schema: z.object({ n: z.number() })
        }
      )
    };
    
    await store.indexTools(tools);
    
    // Search for addition-related tools
    const results = await store.search(["tools"], {
      query: "sum two values",
      limit: 2
    });
    
    expect(results.length).toBeGreaterThan(0);
    expect(results[0].value.tool_id).toBe("add");
  }, 30000);
  
  it("should work with createAgent integration", async () => {
    const tools: ToolRegistry = {
      calculator: tool(
        async ({ a, b }: { a: number; b: number }) => {
          return `Result: ${a + b}`;
        },
        {
          name: "calculator",
          description: "Adds two numbers together",
          schema: z.object({ 
            a: z.number().describe("First number"),
            b: z.number().describe("Second number")
          })
        }
      )
    };
    
    const llm = new ChatOpenAI({ 
      model: "gpt-4o-mini",
      temperature: 0
    });
    
    const agent = await createAgent({
      llm,
      tools,
      store
    });
    
    const result = await agent.invoke({
      messages: [new HumanMessage("Calculate 2 + 2")],
      selected_tool_ids: []
    });
    
    expect(result.messages).toBeDefined();
    expect(result.selected_tool_ids).toContain("calculator");
  }, 60000);
  
  it("should respect TTL settings", async () => {
    // Create store with 1 second TTL for testing
    const shortTTLStore = new RedisVectorBaseStore({
      redisUrl: "redis://localhost:6379",
      embeddings: new OpenAIEmbeddings({
        apiKey: "not-needed",
        configuration: { baseURL: 'http://localhost:8001/v1' }
      }),
      indexName: "test-ttl",
      ttlSeconds: 1
    });
    
    await shortTTLStore.connect();
    
    const tools: ToolRegistry = {
      testTool: tool(
        async () => "test",
        {
          name: "testTool",
          description: "Test tool"
        }
      )
    };
    
    await shortTTLStore.indexTools(tools);
    
    // Check key exists
    const exists = await redisClient.exists("bigtool:tools:meta:testTool");
    expect(exists).toBe(1);
    
    // Wait for TTL to expire
    await new Promise(resolve => setTimeout(resolve, 1500));
    
    // Check key expired
    const expired = await redisClient.exists("bigtool:tools:meta:testTool");
    expect(expired).toBe(0);
    
    await shortTTLStore.disconnect();
  }, 10000);
  
  it("should handle batch operations efficiently", async () => {
    const tools: ToolRegistry = {};
    
    // Create 20 tools
    for (let i = 1; i <= 20; i++) {
      tools[`tool_${i}`] = tool(
        async () => `Result from tool ${i}`,
        {
          name: `tool_${i}`,
          description: `This is tool number ${i} for testing`
        }
      );
    }
    
    const spy = jest.spyOn(console, "log");
    await store.indexTools(tools);
    expect(spy).toHaveBeenCalledWith("Indexed 20 new/updated tools, 0 unchanged");
    
    // Index again - should all be cached
    spy.mockClear();
    await store.indexTools(tools);
    expect(spy).toHaveBeenCalledWith("Indexed 0 new/updated tools, 20 unchanged");
    
    spy.mockRestore();
  }, 30000);
  
  it("should provide accurate index statistics", async () => {
    // Initially empty
    const statsEmpty = await store.getIndexStats();
    expect(statsEmpty.total).toBe(0);
    expect(statsEmpty.oldestIndexed).toBeUndefined();
    expect(statsEmpty.newestIndexed).toBeUndefined();
    
    // Add some tools
    const tools: ToolRegistry = {
      tool1: tool(async () => "1", { name: "tool1", description: "First tool" }),
      tool2: tool(async () => "2", { name: "tool2", description: "Second tool" }),
      tool3: tool(async () => "3", { name: "tool3", description: "Third tool" })
    };
    
    await store.indexTools(tools);
    
    const stats = await store.getIndexStats();
    expect(stats.total).toBe(3);
    expect(stats.oldestIndexed).toBeDefined();
    expect(stats.newestIndexed).toBeDefined();
    expect(stats.oldestIndexed!.getTime()).toBeLessThanOrEqual(stats.newestIndexed!.getTime());
  }, 30000);
  
  it("should handle empty queries gracefully", async () => {
    const tools: ToolRegistry = {
      test: tool(async () => "test", { name: "test", description: "Test tool" })
    };
    
    await store.indexTools(tools);
    
    // Empty query should return empty results
    const results = await store.search(["tools"], {
      query: "",
      limit: 10
    });
    
    expect(results).toEqual([]);
  });
  
  it("should support BaseStore interface methods", async () => {
    const namespace = ["test", "namespace"];
    const key = "testKey";
    const value = {
      tool_id: key,
      name: "Test Tool",
      description: "A tool for testing BaseStore methods"
    };
    
    // Test put
    await store.put(namespace, key, value);
    
    // Test get
    const retrieved = await store.get(namespace, key);
    expect(retrieved).not.toBeNull();
    expect(retrieved!.value.name).toBe(value.name);
    expect(retrieved!.key).toBe(key);
    
    // Test list
    const items = await store.list(namespace);
    expect(items.length).toBeGreaterThan(0);
    expect(items.some(item => item.key === key)).toBe(true);
    
    // Test delete
    await store.delete(namespace, key);
    const deletedItem = await store.get(namespace, key);
    expect(deletedItem).toBeNull();
  }, 30000);
});