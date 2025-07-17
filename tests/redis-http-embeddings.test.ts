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

describe("RedisVectorBaseStore with OpenAIEmbeddings Tests", () => {
  let store: RedisVectorBaseStore;
  let redisClient: RedisClientType;
  let embeddings: OpenAIEmbeddings;
  const testIndexName = "test-http-bigtool-tools";
  
  beforeAll(async () => {
    // Initialize OpenAI embeddings
    embeddings = new OpenAIEmbeddings({
      apiKey: "not-needed",
      configuration: { baseURL: 'http://localhost:8001/v1' }
    });
    
    // Setup Redis client for cleanup
    redisClient = createClient({ url: "redis://localhost:6379" });
    await redisClient.connect();
    
    // Initialize store with HTTP embeddings
    store = new RedisVectorBaseStore({
      redisUrl: "redis://localhost:6379",
      embeddings,
      indexName: testIndexName,
      ttlSeconds: 60, // Short TTL for testing
    });
    
    await store.connect();
  }, 30000);
  
  afterAll(async () => {
    if (!store) return;
    
    // Cleanup
    const keys = await redisClient.keys(`bigtool:*`);
    if (keys.length > 0) {
      await redisClient.del(...keys);
    }
    await store.disconnect();
    await redisClient.quit();
  });
  
  beforeEach(async () => {
    if (!store) return;
    
    // Clear test data before each test
    await store.clearTools();
  });
  
  
  it("should index tools with OpenAI embeddings", async () => {
    if (!store) return;
    
    const tools: ToolRegistry = {
      testTool: tool(
        async () => "test result",
        {
          name: "testTool",
          description: "A test tool for OpenAI embeddings",
          schema: z.object({})
        }
      )
    };
    
    const spy = jest.spyOn(console, "log");
    await store.indexTools(tools);
    expect(spy).toHaveBeenCalledWith("Indexed 1 new/updated tools, 0 unchanged");
    spy.mockRestore();
  }, 30000);
  
  it("should perform semantic search with OpenAI embeddings", async () => {
    if (!store) return;
    
    const tools: ToolRegistry = {
      stringReverse: tool(
        async ({ text }: { text: string }) => text.split('').reverse().join(''),
        {
          name: "stringReverse",
          description: "Reverse the order of characters in a string",
          schema: z.object({ text: z.string() })
        }
      ),
      stringUppercase: tool(
        async ({ text }: { text: string }) => text.toUpperCase(),
        {
          name: "stringUppercase",
          description: "Convert string to uppercase letters",
          schema: z.object({ text: z.string() })
        }
      ),
      numberAdd: tool(
        async ({ a, b }: { a: number; b: number }) => a + b,
        {
          name: "numberAdd",
          description: "Add two numbers together",
          schema: z.object({ a: z.number(), b: z.number() })
        }
      )
    };
    
    await store.indexTools(tools);
    
    // Search for string-related tools
    const results = await store.search(["tools"], {
      query: "manipulate text string",
      limit: 2
    });
    
    expect(results.length).toBe(2);
    // String tools should rank higher than number tools
    const toolIds = results.map(r => r.value.tool_id);
    expect(toolIds).toContain("stringReverse");
    expect(toolIds).toContain("stringUppercase");
  }, 30000);
  
  it("should cache embeddings to avoid redundant calls", async () => {
    if (!store) return;
    
    const tools: ToolRegistry = {
      cachedTool: tool(
        async () => "cached",
        {
          name: "cachedTool",
          description: "Tool to test embedding caching",
          schema: z.object({})
        }
      )
    };
    
    // First indexing - should create embeddings
    const spy = jest.spyOn(console, "log");
    await store.indexTools(tools);
    expect(spy).toHaveBeenCalledWith("Indexed 1 new/updated tools, 0 unchanged");
    
    // Second indexing - should use cache
    spy.mockClear();
    await store.indexTools(tools);
    expect(spy).toHaveBeenCalledWith("Indexed 0 new/updated tools, 1 unchanged");
    
    spy.mockRestore();
  }, 30000);
  
  it("should work with createAgent using OpenAI embeddings", async () => {
    if (!store) return;
    
    const tools: ToolRegistry = {
      concat: tool(
        async ({ a, b }: { a: string; b: string }) => {
          return `${a}${b}`;
        },
        {
          name: "concat",
          description: "Concatenate two strings together",
          schema: z.object({ 
            a: z.string().describe("First string"),
            b: z.string().describe("Second string")
          })
        }
      )
    };
    
    const llm = new ChatOpenAI({ 
      model: "gpt-4o-mini",
      temperature: 0,
      apiKey: process.env.OPENAI_API_KEY
    });
    
    const agent = await createAgent({
      llm,
      tools,
      store
    });
    
    const result = await agent.invoke({
      messages: [new HumanMessage("Concatenate 'Hello' and 'World'")],
      selected_tool_ids: []
    });
    
    expect(result.messages).toBeDefined();
    expect(result.selected_tool_ids).toContain("concat");
  }, 60000);
  
  it("should handle different embedding dimensions correctly", async () => {
    if (!store) return;
    
    // The OpenAI embeddings service should return consistent dimensions
    const testTexts = [
      "Short text",
      "A much longer text with many more words to embed",
      "Special characters: !@#$%^&*()",
      "Numbers: 123456789"
    ];
    
    const embeddingPromises = testTexts.map(text => embeddings.embedQuery(text));
    const results = await Promise.all(embeddingPromises);
    
    // All embeddings should have the same dimension
    const dimension = results[0].length;
    expect(dimension).toBeGreaterThan(0);
    
    results.forEach((embedding, i) => {
      expect(embedding.length).toBe(dimension);
      expect(embedding.every(val => typeof val === 'number')).toBe(true);
    });
  }, 30000);
  
  it("should maintain performance with OpenAI embeddings", async () => {
    if (!store) return;
    
    const tools: ToolRegistry = {};
    
    // Create 10 tools
    for (let i = 1; i <= 10; i++) {
      tools[`tool_${i}`] = tool(
        async () => `Result from tool ${i}`,
        {
          name: `tool_${i}`,
          description: `This is tool number ${i} for performance testing with OpenAI embeddings`
        }
      );
    }
    
    const startTime = Date.now();
    await store.indexTools(tools);
    const indexTime = Date.now() - startTime;
    
    console.log(`Indexed 10 tools with OpenAI embeddings in ${indexTime}ms`);
    
    // Should complete in reasonable time (allowing for network latency)
    expect(indexTime).toBeLessThan(10000); // 10 seconds max
    
    // Test search performance
    const searchStartTime = Date.now();
    const results = await store.search(["tools"], {
      query: "performance testing tool",
      limit: 5
    });
    const searchTime = Date.now() - searchStartTime;
    
    console.log(`Search completed in ${searchTime}ms`);
    expect(searchTime).toBeLessThan(5000); // 5 seconds max
    expect(results.length).toBe(5);
  }, 30000);
  
  it("should handle OpenAI embeddings service errors gracefully", async () => {
    if (!store) return;
    
    // Create a store with invalid embeddings URL
    const badEmbeddings = new OpenAIEmbeddings({
      apiKey: "not-needed",
      configuration: { baseURL: 'http://localhost:9999/v1' } // Non-existent service
    });
    
    const badStore = new RedisVectorBaseStore({
      redisUrl: "redis://localhost:6379",
      embeddings: badEmbeddings,
      indexName: "test-bad-embeddings",
      ttlSeconds: 60
    });
    
    await badStore.connect();
    
    const tools: ToolRegistry = {
      errorTool: tool(
        async () => "error",
        {
          name: "errorTool",
          description: "Tool to test error handling"
        }
      )
    };
    
    // Should throw an error when trying to index
    await expect(badStore.indexTools(tools)).rejects.toThrow();
    
    await badStore.disconnect();
  });
});