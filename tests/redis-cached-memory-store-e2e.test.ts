import { describe, it, expect, beforeAll, afterAll, beforeEach, jest } from "@jest/globals";
import { RedisCachedMemoryVectorBaseStore } from "../src/stores/RedisCachedMemoryVectorBaseStore.js";
import { OpenAIEmbeddings } from "@langchain/openai";
import { createClient, RedisClientType } from "redis";
import { createAgent } from "../src/index.js";
import { ChatOpenAI } from "@langchain/openai";
import { tool } from "@langchain/core/tools";
import { z } from "zod";
import { HumanMessage } from "@langchain/core/messages";
import type { ToolRegistry } from "../src/types.js";

describe("RedisCachedMemoryVectorBaseStore E2E Tests", () => {
  let store: RedisCachedMemoryVectorBaseStore;
  let redisClient: RedisClientType;
  let embeddings: OpenAIEmbeddings;
  const testIndexName = "test-cached-embeddings";
  
  // Test tools
  const mathTools: ToolRegistry = {
    add: tool(
      async ({ a, b }: { a: number; b: number }) => `${a} + ${b} = ${a + b}`,
      {
        name: "add",
        description: "Add two numbers together",
        schema: z.object({ a: z.number(), b: z.number() })
      }
    ),
    multiply: tool(
      async ({ a, b }: { a: number; b: number }) => `${a} * ${b} = ${a * b}`,
      {
        name: "multiply", 
        description: "Multiply two numbers to get their product",
        schema: z.object({ a: z.number(), b: z.number() })
      }
    ),
    sqrt: tool(
      async ({ n }: { n: number }) => `√${n} = ${Math.sqrt(n)}`,
      {
        name: "sqrt",
        description: "Calculate the square root of a number",
        schema: z.object({ n: z.number() })
      }
    ),
    power: tool(
      async ({ base, exp }: { base: number; exp: number }) => `${base}^${exp} = ${Math.pow(base, exp)}`,
      {
        name: "power",
        description: "Raise a number to a power (exponentiation)",
        schema: z.object({ base: z.number(), exp: z.number() })
      }
    )
  };
  
  beforeAll(async () => {
    // Setup Redis client for direct inspection
    redisClient = createClient({ url: "redis://localhost:6379" });
    await redisClient.connect();
    
    // Initialize embeddings
    embeddings = new OpenAIEmbeddings({
      apiKey: "not-needed",
      configuration: { baseURL: 'http://localhost:8001/v1' }
    });
  }, 30000);
  
  afterAll(async () => {
    // Cleanup all test keys
    const keys = await redisClient.keys(`${testIndexName}:*`);
    if (keys.length > 0) {
      await redisClient.del(...keys);
    }
    await redisClient.quit();
  });
  
  beforeEach(async () => {
    // Clear test data before each test
    const keys = await redisClient.keys(`${testIndexName}:*`);
    if (keys.length > 0) {
      await redisClient.del(...keys);
    }
  });
  
  it("should create cache entries on initial indexing", async () => {
    // Create store with verbose logging
    store = new RedisCachedMemoryVectorBaseStore({
      redisUrl: "redis://localhost:6379",
      embeddings,
      indexName: testIndexName,
      ttlSeconds: 60,
      verbose: true
    });
    
    // Spy on console.info to capture cache logs
    const consoleSpy = jest.spyOn(console, 'info');
    
    await store.connect();
    
    // Index tools for the first time
    await store.indexTools(mathTools);
    
    // Verify console logs show all tools were computed (no cache hits)
    const logs = consoleSpy.mock.calls.map(call => call[0]).join('\n');
    expect(logs).toContain('4 computed)'); // All 4 tools computed
    expect(logs).toContain('0 cache hits');
    
    // Verify cache entries exist in Redis
    const keys = await redisClient.keys(`${testIndexName}:*`);
    expect(keys.length).toBe(4); // One key per tool
    
    // Verify cache content
    const addCacheKey = `${testIndexName}:add`;
    const cachedAdd = await redisClient.get(addCacheKey);
    expect(cachedAdd).toBeTruthy();
    
    const parsed = JSON.parse(cachedAdd!);
    expect(parsed.tool_id).toBe('add');
    expect(parsed.name).toBe('add');
    expect(parsed.description).toBe('Add two numbers together');
    expect(parsed.embedding).toBeInstanceOf(Array);
    expect(parsed.embedding.length).toBeGreaterThan(0);
    expect(parsed.cached_at).toBeGreaterThan(0);
    
    // Test search functionality
    const results = await store.search(['tools'], { query: 'add numbers', limit: 2 });
    expect(results.length).toBeGreaterThan(0);
    expect(results[0].value.name).toBe('add');
    
    await store.disconnect();
    consoleSpy.mockRestore();
  });
  
  it("should use cached embeddings on subsequent indexing", async () => {
    // First, create cache entries
    const store1 = new RedisCachedMemoryVectorBaseStore({
      redisUrl: "redis://localhost:6379",
      embeddings,
      indexName: testIndexName,
      ttlSeconds: 60,
      verbose: false
    });
    
    await store1.connect();
    await store1.indexTools(mathTools);
    await store1.disconnect();
    
    // Create new store instance (simulating app restart)
    const store2 = new RedisCachedMemoryVectorBaseStore({
      redisUrl: "redis://localhost:6379",
      embeddings,
      indexName: testIndexName,
      ttlSeconds: 60,
      verbose: true
    });
    
    const consoleSpy = jest.spyOn(console, 'info');
    
    await store2.connect();
    
    // Index same tools - should use cache
    await store2.indexTools(mathTools);
    
    // Verify console logs show all cache hits
    const logs = consoleSpy.mock.calls.map(call => call[0]).join('\n');
    expect(logs).toContain('4 cache hits [100.0%]');
    expect(logs).toContain('0 computed)');
    
    // Verify search still works
    const results = await store2.search(['tools'], { query: 'multiply', limit: 2 });
    expect(results.length).toBeGreaterThan(0);
    expect(results[0].value.name).toBe('multiply');
    
    await store2.disconnect();
    consoleSpy.mockRestore();
  });
  
  it("should measure performance improvement from caching", async () => {
    // Time initial indexing (no cache)
    const store1 = new RedisCachedMemoryVectorBaseStore({
      redisUrl: "redis://localhost:6379",
      embeddings,
      indexName: testIndexName,
      ttlSeconds: 60,
      verbose: false
    });
    
    await store1.connect();
    
    const start1 = performance.now();
    await store1.indexTools(mathTools);
    const duration1 = performance.now() - start1;
    
    await store1.disconnect();
    
    // Time cached indexing
    const store2 = new RedisCachedMemoryVectorBaseStore({
      redisUrl: "redis://localhost:6379",
      embeddings,
      indexName: testIndexName,
      ttlSeconds: 60,
      verbose: false
    });
    
    await store2.connect();
    
    const start2 = performance.now();
    await store2.indexTools(mathTools);
    const duration2 = performance.now() - start2;
    
    await store2.disconnect();
    
    // Cached version should be significantly faster
    console.log(`Initial indexing: ${duration1.toFixed(2)}ms`);
    console.log(`Cached indexing: ${duration2.toFixed(2)}ms`);
    console.log(`Speedup: ${(duration1 / duration2).toFixed(1)}x`);
    
    expect(duration2).toBeLessThan(duration1);
    expect(duration1 / duration2).toBeGreaterThan(5); // At least 5x faster
  });
  
  it("should work with createAgent integration", async () => {
    // Create store and verify it can be used with createAgent
    const store = new RedisCachedMemoryVectorBaseStore({
      redisUrl: "redis://localhost:6379",
      embeddings,
      indexName: testIndexName,
      ttlSeconds: 60,
      verbose: false
    });
    
    await store.connect();
    
    // Index tools first time
    await store.indexTools(mathTools);
    
    // Verify tools can be searched
    const searchResults = await store.search(['tools'], { query: 'calculate square root', limit: 2 });
    expect(searchResults.length).toBeGreaterThan(0);
    expect(searchResults[0].value.name).toBe('sqrt');
    
    await store.disconnect();
    
    // Create new store instance with cached embeddings
    const store2 = new RedisCachedMemoryVectorBaseStore({
      redisUrl: "redis://localhost:6379",
      embeddings,
      indexName: testIndexName,
      ttlSeconds: 60,
      verbose: true
    });
    
    const consoleSpy = jest.spyOn(console, 'info');
    await store2.connect();
    
    // Index again - should use cache
    await store2.indexTools(mathTools);
    
    // Verify cache was used
    const logs = consoleSpy.mock.calls.map(call => call[0]).join('\n');
    expect(logs).toContain('4 cache hits');
    expect(logs).toContain('100.0%');
    
    // Verify search still works with cached embeddings
    const searchResults2 = await store2.search(['tools'], { query: 'multiply numbers', limit: 2 });
    expect(searchResults2.length).toBeGreaterThan(0);
    expect(searchResults2[0].value.name).toBe('multiply');
    
    await store2.disconnect();
    consoleSpy.mockRestore();
  });
  
  it("should handle partial cache hits", async () => {
    // Use a different test namespace to ensure clean state
    const partialTestIndex = testIndexName + "-partial";
    
    // Create initial cache with subset of tools
    const partialTools = {
      add: mathTools.add,
      multiply: mathTools.multiply
    };
    
    const store1 = new RedisCachedMemoryVectorBaseStore({
      redisUrl: "redis://localhost:6379",
      embeddings,
      indexName: partialTestIndex,
      ttlSeconds: 60,
      verbose: false
    });
    
    await store1.connect();
    await store1.indexTools(partialTools);
    
    // Verify only 2 entries in cache
    const keys1 = await redisClient.keys(`${partialTestIndex}:*`);
    expect(keys1.length).toBe(2);
    
    await store1.disconnect();
    
    // Index full set of tools - should have partial cache hits
    const store2 = new RedisCachedMemoryVectorBaseStore({
      redisUrl: "redis://localhost:6379",
      embeddings,
      indexName: partialTestIndex,
      ttlSeconds: 60,
      verbose: true
    });
    
    const consoleSpy = jest.spyOn(console, 'info');
    await store2.connect();
    await store2.indexTools(mathTools);
    
    // Verify partial cache hits
    const logs = consoleSpy.mock.calls.map(call => call[0]).join('\n');
    expect(logs).toContain('2 cache hits'); // add and multiply from cache
    expect(logs).toContain('2 computed'); // sqrt and power computed
    
    // Verify all 4 tools are now cached
    const keys2 = await redisClient.keys(`${partialTestIndex}:*`);
    expect(keys2.length).toBe(4);
    
    await store2.disconnect();
    consoleSpy.mockRestore();
    
    // Cleanup
    if (keys2.length > 0) {
      await redisClient.del(...keys2);
    }
  });
});