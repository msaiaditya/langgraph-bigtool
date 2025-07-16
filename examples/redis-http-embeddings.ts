#!/usr/bin/env node
import { createAgent } from "../src/index.js";
import { RedisStore } from "../src/stores/RedisStore.js";
import { HTTPEmbeddings } from "../src/embeddings/index.js";
import { ChatOpenAI } from "@langchain/openai";
import { tool } from "@langchain/core/tools";
import { z } from "zod";
import { HumanMessage } from "@langchain/core/messages";
import type { ToolRegistry } from "../src/types.js";

// Create string manipulation tools
const stringTools: ToolRegistry = {
  uppercase: tool(
    async ({ text }: { text: string }) => {
      console.log(`Converting "${text}" to uppercase`);
      return text.toUpperCase();
    },
    {
      name: "uppercase",
      description: "Convert text to uppercase letters",
      schema: z.object({
        text: z.string().describe("Text to convert to uppercase")
      })
    }
  ),
  
  lowercase: tool(
    async ({ text }: { text: string }) => {
      console.log(`Converting "${text}" to lowercase`);
      return text.toLowerCase();
    },
    {
      name: "lowercase",
      description: "Convert text to lowercase letters",
      schema: z.object({
        text: z.string().describe("Text to convert to lowercase")
      })
    }
  ),
  
  reverse: tool(
    async ({ text }: { text: string }) => {
      console.log(`Reversing "${text}"`);
      return text.split('').reverse().join('');
    },
    {
      name: "reverse",
      description: "Reverse the order of characters in text",
      schema: z.object({
        text: z.string().describe("Text to reverse")
      })
    }
  ),
  
  countWords: tool(
    async ({ text }: { text: string }) => {
      console.log(`Counting words in "${text}"`);
      const words = text.trim().split(/\s+/).filter(word => word.length > 0);
      return `The text contains ${words.length} word(s)`;
    },
    {
      name: "countWords",
      description: "Count the number of words in text",
      schema: z.object({
        text: z.string().describe("Text to count words in")
      })
    }
  ),
  
  replaceText: tool(
    async ({ text, find, replace }: { text: string; find: string; replace: string }) => {
      console.log(`Replacing "${find}" with "${replace}" in "${text}"`);
      return text.replaceAll(find, replace);
    },
    {
      name: "replaceText",
      description: "Replace all occurrences of a substring with another substring",
      schema: z.object({
        text: z.string().describe("Text to search in"),
        find: z.string().describe("Text to find"),
        replace: z.string().describe("Text to replace with")
      })
    }
  ),
  
  extractNumbers: tool(
    async ({ text }: { text: string }) => {
      console.log(`Extracting numbers from "${text}"`);
      const numbers = text.match(/\d+/g) || [];
      return `Found numbers: ${numbers.join(', ') || 'none'}`;
    },
    {
      name: "extractNumbers",
      description: "Extract all numbers from text",
      schema: z.object({
        text: z.string().describe("Text to extract numbers from")
      })
    }
  )
};

async function main() {
  console.log("🚀 Redis Store with HTTP Embeddings Demo\n");
  
  // Initialize HTTP embeddings
  const embeddings = new HTTPEmbeddings({
    serviceUrl: process.env.EMBEDDINGS_SERVICE_URL || 'http://localhost:8001'
  });
  
  // Check if embeddings service is available
  console.log("🔍 Checking embeddings service health...");
  const isHealthy = await embeddings.checkHealth();
  
  if (!isHealthy) {
    console.error("\n⚠️  Embeddings service is not available!");
    console.error("   Please ensure the embeddings service is running at http://localhost:8001");
    console.error("   See the embeddings-service directory for setup instructions.");
    console.error("\n   To start the service:");
    console.error("   cd embeddings-service && docker-compose up -d\n");
    return;
  }
  
  console.log("✅ Embeddings service is healthy\n");
  
  // Initialize Redis store with HTTP embeddings
  const store = new RedisStore({
    redisUrl: process.env.REDIS_URL || "redis://localhost:6379",
    embeddings,
    indexName: "bigtool-string-tools",
    ttlSeconds: 7 * 24 * 60 * 60, // 7 days
    verbose: true
  });
  
  try {
    // Connect to Redis
    await store.connect();
    console.log("\n📊 Redis connection established");
    
    // Clear previous data for clean demo
    await store.clearTools();
    console.log("🧹 Cleared previous tool data");
    
    // Get index stats before
    const statsBefore = await store.getIndexStats();
    console.log(`\n📈 Index stats before: ${statsBefore.total} tools indexed`);
    
    // Create agent with Redis store and HTTP embeddings
    const llm = new ChatOpenAI({ 
      model: "gpt-4o-mini",
      temperature: 0,
      apiKey: process.env.OPENAI_API_KEY
    });
    
    console.log("\n🔧 Creating agent with string manipulation tools...");
    const agent = await createAgent({
      llm,
      tools: stringTools,
      store,
      options: {
        limit: 2 // Retrieve up to 2 tools at a time
      }
    });
    
    // Get index stats after
    const statsAfter = await store.getIndexStats();
    console.log(`\n📈 Index stats after: ${statsAfter.total} tools indexed`);
    if (statsAfter.newestIndexed) {
      console.log(`   Latest indexing: ${statsAfter.newestIndexed.toLocaleString()}`);
    }
    
    // Test 1: Text transformation
    console.log("\n--- Test 1: Text Transformation ---");
    const result1 = await agent.invoke({
      messages: [new HumanMessage("Convert 'Hello World' to uppercase")],
      selected_tool_ids: []
    });
    
    console.log("\nAgent response:", result1.messages[result1.messages.length - 1].content);
    console.log("Tools used:", result1.selected_tool_ids);
    
    // Test 2: Text analysis
    console.log("\n--- Test 2: Text Analysis ---");
    const result2 = await agent.invoke({
      messages: [new HumanMessage("Count the words in 'The quick brown fox jumps over the lazy dog'")],
      selected_tool_ids: []
    });
    
    console.log("\nAgent response:", result2.messages[result2.messages.length - 1].content);
    console.log("Tools used:", result2.selected_tool_ids);
    
    // Test 3: Complex string operation
    console.log("\n--- Test 3: Complex String Operation ---");
    const result3 = await agent.invoke({
      messages: [new HumanMessage("Replace all spaces with underscores in 'Hello World from Redis' and then reverse it")],
      selected_tool_ids: []
    });
    
    console.log("\nAgent response:", result3.messages[result3.messages.length - 1].content);
    console.log("Tools used:", result3.selected_tool_ids);
    
    // Test 4: Number extraction
    console.log("\n--- Test 4: Number Extraction ---");
    const result4 = await agent.invoke({
      messages: [new HumanMessage("Extract all numbers from 'I have 5 apples, 3 oranges, and 10 bananas'")],
      selected_tool_ids: []
    });
    
    console.log("\nAgent response:", result4.messages[result4.messages.length - 1].content);
    console.log("Tools used:", result4.selected_tool_ids);
    
    // Demonstrate caching with HTTP embeddings
    console.log("\n--- Demonstrating Caching with HTTP Embeddings ---");
    console.log("Creating agent again with same tools...");
    
    const agent2 = await createAgent({
      llm,
      tools: stringTools,
      store,
      options: {
        limit: 2
      }
    });
    
    console.log("✅ Second agent creation completed (should use cached embeddings)");
    
    // Show final stats
    const finalStats = await store.getIndexStats();
    console.log(`\n📊 Final index stats: ${finalStats.total} tools indexed`);
    
    // Test semantic search directly
    console.log("\n--- Testing Semantic Search ---");
    const searchResults = await store.search(["tools"], {
      query: "transform text case",
      limit: 3
    });
    
    console.log("Search results for 'transform text case':");
    searchResults.forEach((result, i) => {
      console.log(`  ${i + 1}. ${result.value.name} (score: ${result.score?.toFixed(4)})`);
    });
    
  } catch (error) {
    console.error("\n❌ Error:", error);
    console.error("\n💡 Make sure both Redis and the embeddings service are running:");
    console.error("   1. Redis: docker-compose up -d redis");
    console.error("   2. Embeddings: cd embeddings-service && docker-compose up -d");
  } finally {
    // Disconnect from Redis
    await store.disconnect();
    console.log("\n👋 Redis connection closed");
  }
}

// Run the demo
main().catch(console.error);