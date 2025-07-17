import { ChatOpenAI } from "@langchain/openai";
import { OpenAIEmbeddings } from "@langchain/openai";
import { HumanMessage } from "@langchain/core/messages";
import { DynamicTool } from "@langchain/core/tools";
import { createAgent, RedisCachedMemoryVectorBaseStore } from "../src/index.js";

/**
 * Example demonstrating RedisCachedMemoryVectorBaseStore usage.
 * 
 * This store caches computed embeddings in Redis and loads them into memory
 * for fast semantic search. Embeddings are computed only once and reused
 * across application restarts.
 * 
 * Run with: tsx examples/redis-cached-memory.ts
 * Make sure Redis is running: docker run -p 6379:6379 redis:latest
 */

// Create a variety of math-related tools
const mathTools = {
  add: new DynamicTool({
    name: "add",
    description: "Add two numbers together",
    func: async ({ a, b }: any) => `${a} + ${b} = ${a + b}`
  }),
  
  subtract: new DynamicTool({
    name: "subtract",
    description: "Subtract one number from another",
    func: async ({ a, b }: any) => `${a} - ${b} = ${a - b}`
  }),
  
  multiply: new DynamicTool({
    name: "multiply",
    description: "Multiply two numbers together",
    func: async ({ a, b }: any) => `${a} * ${b} = ${a * b}`
  }),
  
  divide: new DynamicTool({
    name: "divide",
    description: "Divide one number by another",
    func: async ({ a, b }: any) => {
      if (b === 0) return "Error: Division by zero";
      return `${a} / ${b} = ${a / b}`;
    }
  }),
  
  power: new DynamicTool({
    name: "power",
    description: "Raise a number to a power (exponentiation)",
    func: async ({ base, exponent }: any) => `${base}^${exponent} = ${Math.pow(base, exponent)}`
  }),
  
  sqrt: new DynamicTool({
    name: "sqrt",
    description: "Calculate the square root of a number",
    func: async ({ n }: any) => `√${n} = ${Math.sqrt(n)}`
  }),
  
  factorial: new DynamicTool({
    name: "factorial",
    description: "Calculate the factorial of a positive integer",
    func: async ({ n }: any) => {
      if (n < 0) return "Error: Factorial is not defined for negative numbers";
      let result = 1;
      for (let i = 2; i <= n; i++) result *= i;
      return `${n}! = ${result}`;
    }
  }),
  
  percentage: new DynamicTool({
    name: "percentage",
    description: "Calculate what percentage one number is of another",
    func: async ({ part, whole }: any) => `${part} is ${(part / whole * 100).toFixed(2)}% of ${whole}`
  })
};

async function main() {
  console.log("🚀 Redis Cached Memory Vector Store Example\n");
  
  // Initialize embeddings and LLM
  const embeddings = new OpenAIEmbeddings({
    model: "text-embedding-3-small",
    apiKey: process.env.OPENAI_API_KEY
  });
  
  const llm = new ChatOpenAI({
    model: "gpt-4o-mini",
    temperature: 0,
    apiKey: process.env.OPENAI_API_KEY
  });
  
  // Create the Redis-cached memory store
  const store = new RedisCachedMemoryVectorBaseStore({
    redisUrl: process.env.REDIS_URL || "redis://localhost:6379",
    embeddings,
    ttlSeconds: 7 * 24 * 60 * 60, // 7 days
    verbose: true
  });
  
  // Connect to Redis
  await store.connect();
  console.log("✅ Connected to Redis\n");
  
  // Create agent with tools
  const agent = await createAgent({
    llm,
    tools: mathTools,
    store,
    maxToolRoundtrips: 2
  });
  
  // Test 1: First run - embeddings will be computed and cached
  console.log("=== Test 1: Initial Run (Computing Embeddings) ===");
  const result1 = await agent.invoke({
    messages: [new HumanMessage("I need to calculate 15% of 200")]
  });
  
  console.log("Result:", result1.messages[result1.messages.length - 1].content);
  console.log();
  
  // Test 2: Second run - embeddings loaded from cache
  console.log("=== Test 2: Cache Hit Run (Using Cached Embeddings) ===");
  
  // Create a new store instance to simulate application restart
  const store2 = new RedisCachedMemoryVectorBaseStore({
    redisUrl: process.env.REDIS_URL || "redis://localhost:6379",
    embeddings,
    ttlSeconds: 7 * 24 * 60 * 60,
    verbose: true
  });
  
  await store2.connect();
  
  const agent2 = await createAgent({
    llm,
    tools: mathTools,
    store: store2,
    maxToolRoundtrips: 2
  });
  
  const result2 = await agent2.invoke({
    messages: [new HumanMessage("What's the square root of 144?")]
  });
  
  console.log("Result:", result2.messages[result2.messages.length - 1].content);
  console.log();
  
  // Test 3: Complex calculation requiring multiple tools
  console.log("=== Test 3: Complex Calculation ===");
  const result3 = await agent2.invoke({
    messages: [new HumanMessage("Calculate (5^3) + (12 * 8) - 50")]
  });
  
  console.log("Result:", result3.messages[result3.messages.length - 1].content);
  
  // Cleanup
  await store.disconnect();
  await store2.disconnect();
  console.log("\n✅ Disconnected from Redis");
}

// Run the example
main().catch(console.error);