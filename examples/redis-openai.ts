#!/usr/bin/env node
import { createAgent } from "../src/index.js";
import { RedisVectorBaseStore } from "../src/stores/RedisVectorBaseStore.js";
import { OpenAIEmbeddings } from "@langchain/openai";
import { ChatOpenAI } from "@langchain/openai";
import { tool } from "@langchain/core/tools";
import { z } from "zod";
import { HumanMessage } from "@langchain/core/messages";
import type { ToolRegistry } from "../src/types.js";

// Create math tools
const mathTools: ToolRegistry = {
  add: tool(
    async ({ a, b }: { a: number; b: number }) => {
      console.log(`Adding ${a} + ${b}`);
      return `${a} + ${b} = ${a + b}`;
    },
    {
      name: "add",
      description: "Add two numbers together",
      schema: z.object({
        a: z.number().describe("First number"),
        b: z.number().describe("Second number")
      })
    }
  ),
  
  multiply: tool(
    async ({ a, b }: { a: number; b: number }) => {
      console.log(`Multiplying ${a} * ${b}`);
      return `${a} * ${b} = ${a * b}`;
    },
    {
      name: "multiply",
      description: "Multiply two numbers",
      schema: z.object({
        a: z.number().describe("First number"),
        b: z.number().describe("Second number")
      })
    }
  ),
  
  sqrt: tool(
    async ({ n }: { n: number }) => {
      console.log(`Calculating square root of ${n}`);
      if (n < 0) {
        return `Cannot calculate square root of negative number ${n}`;
      }
      return `√${n} = ${Math.sqrt(n)}`;
    },
    {
      name: "sqrt",
      description: "Calculate the square root of a number",
      schema: z.object({
        n: z.number().describe("Number to find square root of")
      })
    }
  ),
  
  power: tool(
    async ({ base, exponent }: { base: number; exponent: number }) => {
      console.log(`Calculating ${base}^${exponent}`);
      return `${base}^${exponent} = ${Math.pow(base, exponent)}`;
    },
    {
      name: "power",
      description: "Raise a number to a power (exponentiation)",
      schema: z.object({
        base: z.number().describe("Base number"),
        exponent: z.number().describe("Exponent")
      })
    }
  ),
  
  factorial: tool(
    async ({ n }: { n: number }) => {
      console.log(`Calculating factorial of ${n}`);
      if (n < 0 || !Number.isInteger(n)) {
        return `Factorial is only defined for non-negative integers`;
      }
      let result = 1;
      for (let i = 2; i <= n; i++) {
        result *= i;
      }
      return `${n}! = ${result}`;
    },
    {
      name: "factorial",
      description: "Calculate the factorial of a non-negative integer",
      schema: z.object({
        n: z.number().describe("Non-negative integer")
      })
    }
  )
};

async function main() {
  console.log("🚀 Redis Vector Base Store with OpenAI Embeddings Demo\n");
  
  // Initialize embeddings and store
  const embeddings = new OpenAIEmbeddings({
    model: "text-embedding-3-small"
  });
  
  const store = new RedisVectorBaseStore({
    redisUrl: process.env.REDIS_URL || "redis://localhost:6379",
    embeddings,
    indexName: "bigtool-math-tools",
    ttlSeconds: 7 * 24 * 60 * 60, // 7 days
    verbose: true
  });
  
  try {
    // Connect to Redis
    await store.connect();
    console.log("\n📊 Redis connection established");
    
    // Get index stats before
    const statsBefore = await store.getIndexStats();
    console.log(`\n📈 Index stats before: ${statsBefore.total} tools indexed`);
    
    // Create agent with Redis store
    const llm = new ChatOpenAI({ 
      model: "gpt-4o-mini",
      temperature: 0
    });
    
    console.log("\n🔧 Creating agent with math tools...");
    const agent = await createAgent({
      llm,
      tools: mathTools,
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
    
    // Test 1: Square root calculation
    console.log("\n--- Test 1: Square Root ---");
    const result1 = await agent.invoke({
      messages: [new HumanMessage("What's the square root of 144?")],
      selected_tool_ids: []
    });
    
    console.log("\nAgent response:", result1.messages[result1.messages.length - 1].content);
    console.log("Tools used:", result1.selected_tool_ids);
    
    // Test 2: Complex calculation
    console.log("\n--- Test 2: Complex Calculation ---");
    const result2 = await agent.invoke({
      messages: [new HumanMessage("Calculate 5 factorial and then find the square root of the result")],
      selected_tool_ids: []
    });
    
    console.log("\nAgent response:", result2.messages[result2.messages.length - 1].content);
    console.log("Tools used:", result2.selected_tool_ids);
    
    // Test 3: Multiple operations
    console.log("\n--- Test 3: Multiple Operations ---");
    const result3 = await agent.invoke({
      messages: [new HumanMessage("I need to add 15 and 27, then multiply the result by 3")],
      selected_tool_ids: []
    });
    
    console.log("\nAgent response:", result3.messages[result3.messages.length - 1].content);
    console.log("Tools used:", result3.selected_tool_ids);
    
    // Demonstrate caching - run the same agent creation again
    console.log("\n--- Demonstrating Caching ---");
    console.log("Creating agent again with same tools...");
    
    const agent2 = await createAgent({
      llm,
      tools: mathTools,
      store,
      options: {
        limit: 2
      }
    });
    
    console.log("✅ Second agent creation completed (should use cached embeddings)");
    
    // Show final stats
    const finalStats = await store.getIndexStats();
    console.log(`\n📊 Final index stats: ${finalStats.total} tools indexed`);
    
  } catch (error) {
    console.error("\n❌ Error:", error);
    console.error("\n💡 Make sure Redis is running:");
    console.error("   docker run -d -p 6379:6379 redis/redis-stack:latest");
  } finally {
    // Disconnect from Redis
    await store.disconnect();
    console.log("\n👋 Redis connection closed");
  }
}

// Run the demo
main().catch(console.error);