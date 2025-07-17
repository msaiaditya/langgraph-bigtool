#!/usr/bin/env node
/**
 * Quick start example showing Redis with HTTP embeddings
 * This demonstrates the most basic setup without OpenAI dependency for embeddings
 */

import { RedisVectorBaseStore } from "../src/stores/RedisVectorBaseStore.js";
import { OpenAIEmbeddings } from "@langchain/openai";
import { tool } from "@langchain/core/tools";
import { z } from "zod";
import type { ToolRegistry } from "../src/types.js";

async function main() {
  console.log("🚀 Redis + HTTP Embeddings Quick Start\n");
  
  // 1. Initialize OpenAI embeddings with custom URL (no API key needed!)
  const embeddings = new OpenAIEmbeddings({
    apiKey: "not-needed",
    configuration: {
      baseURL: 'http://localhost:8001/v1'
    }
  });
  
  // 3. Create Redis store with HTTP embeddings
  const store = new RedisVectorBaseStore({
    redisUrl: "redis://localhost:6379",
    embeddings,
    indexName: "quickstart-tools",
    verbose: true
  });
  
  await store.connect();
  console.log("✅ Connected to Redis\n");
  
  // 4. Create some simple tools
  const tools: ToolRegistry = {
    greet: tool(
      async ({ name }: { name: string }) => `Hello, ${name}!`,
      {
        name: "greet",
        description: "Greet someone by name",
        schema: z.object({ name: z.string() })
      }
    ),
    
    calculate: tool(
      async ({ expression }: { expression: string }) => {
        // Simple calculator (don't use eval in production!)
        try {
          return `Result: ${eval(expression)}`;
        } catch (e) {
          return "Invalid expression";
        }
      },
      {
        name: "calculate",
        description: "Calculate simple math expressions",
        schema: z.object({ expression: z.string() })
      }
    )
  };
  
  // 5. Index the tools
  console.log("📝 Indexing tools...");
  await store.indexTools(tools);
  
  // 6. Test semantic search
  console.log("\n🔍 Testing semantic search:");
  
  const queries = [
    "say hello to someone",
    "math calculation",
    "greeting message"
  ];
  
  for (const query of queries) {
    const results = await store.search(["tools"], { query, limit: 1 });
    if (results.length > 0) {
      console.log(`   Query: "${query}" → Found: ${results[0].value.name}`);
    }
  }
  
  // 7. Show caching in action
  console.log("\n♻️  Testing caching:");
  await store.indexTools(tools); // Should use cache
  
  // 8. Get statistics
  const stats = await store.getIndexStats();
  console.log(`\n📊 Statistics: ${stats.total} tools indexed`);
  
  await store.disconnect();
  console.log("\n✅ Done! Redis connection closed.");
}

main().catch(console.error);