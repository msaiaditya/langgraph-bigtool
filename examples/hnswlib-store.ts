/**
 * Example of using HNSWLibStore for high-performance vector search.
 * 
 * This example demonstrates:
 * - Creating a HNSWLibStore with HTTP embeddings
 * - Using the store with createAgent
 * - Tool discovery through semantic search
 */

import { createAgent, HNSWLibStore, createToolRegistry, HTTPEmbeddings } from "../src/index.js";
import { ChatOpenAI } from "@langchain/openai";
import { tool } from "@langchain/core/tools";
import { z } from "zod";

// Create some example tools
const weatherTool = tool(
  async ({ city }) => {
    const weather = ["sunny", "cloudy", "rainy"][Math.floor(Math.random() * 3)];
    const temp = Math.floor(Math.random() * 30) + 10;
    return `The weather in ${city} is ${weather} with a temperature of ${temp}°C`;
  },
  {
    name: "get_weather",
    description: "Get the current weather for a city. Returns temperature and conditions.",
    schema: z.object({
      city: z.string().describe("The city name")
    })
  }
);

const calculatorTool = tool(
  async ({ a, b, operation }) => {
    switch (operation) {
      case "add": return `${a} + ${b} = ${a + b}`;
      case "subtract": return `${a} - ${b} = ${a - b}`;
      case "multiply": return `${a} * ${b} = ${a * b}`;
      case "divide": return b !== 0 ? `${a} / ${b} = ${a / b}` : "Error: Division by zero";
      default: return "Error: Unknown operation";
    }
  },
  {
    name: "calculator",
    description: "Perform basic arithmetic operations (add, subtract, multiply, divide) on two numbers",
    schema: z.object({
      a: z.number().describe("The first number"),
      b: z.number().describe("The second number"),
      operation: z.enum(["add", "subtract", "multiply", "divide"]).describe("The operation to perform")
    })
  }
);

const timeTool = tool(
  async ({ timezone }) => {
    const now = new Date();
    const formatter = new Intl.DateTimeFormat('en-US', {
      timeZone: timezone,
      dateStyle: 'full',
      timeStyle: 'long'
    });
    return `Current time in ${timezone}: ${formatter.format(now)}`;
  },
  {
    name: "get_time",
    description: "Get the current time in a specific timezone",
    schema: z.object({
      timezone: z.string().describe("The timezone (e.g., 'America/New_York', 'Europe/London')")
    })
  }
);

async function main() {
  console.log("=== HNSWLib Store Example ===\n");
  
  // Create tool registry
  const toolRegistry = createToolRegistry([weatherTool, calculatorTool, timeTool]);
  
  // Create embeddings instance
  console.log("1. Creating HTTPEmbeddings...");
  const embeddings = new HTTPEmbeddings({ serviceUrl: 'http://localhost:8001' });
  
  // Check if embeddings service is healthy
  const isHealthy = await embeddings.checkHealth();
  console.log(`   Embeddings service health: ${isHealthy ? '✓ Online' : '✗ Offline'}`);
  console.log(`   Service URL: ${embeddings.getServiceUrl()}\n`);
  
  if (!isHealthy) {
    console.error("⚠️  Embeddings service is not available!");
    console.error("   Please ensure the embeddings service is running at", embeddings.getServiceUrl());
    console.error("   See the embeddings-service directory for setup instructions.\n");
    return;
  }
  
  // Create HNSWLibStore with embeddings
  console.log("2. Creating HNSWLibStore with HNSW configuration...");
  const store = new HNSWLibStore(embeddings, {
    space: 'cosine',        // Similarity metric
    numDimensions: 384      // all-MiniLM-L6-v2 dimensions
  });
  console.log("   ✓ Store created with cosine similarity");
  
  // Create the agent with HNSWLib store
  console.log("\n3. Creating agent with HNSWLib store...");
  const llm = new ChatOpenAI({ 
    model: "gpt-4",
    temperature: 0 
  });
  
  const agent = await createAgent({
    llm,
    tools: toolRegistry,
    store,
    options: {
      limit: 2 // Retrieve up to 2 tools per search
    }
  });
  console.log("   ✓ Agent created with automatic tool indexing\n");
  
  // Test 1: Weather query
  console.log("4. Testing weather query...");
  const weatherResult = await agent.invoke({
    messages: [{ role: "user", content: "What's the weather like in Tokyo?" }],
    selected_tool_ids: []
  });
  
  const weatherMessage = weatherResult.messages[weatherResult.messages.length - 1];
  console.log(`   Query: "What's the weather like in Tokyo?"`);
  console.log(`   Retrieved tools: ${weatherResult.selected_tool_ids.join(', ')}`);
  console.log(`   Response: ${weatherMessage.content}\n`);
  
  // Test 2: Math query
  console.log("5. Testing math query...");
  const mathResult = await agent.invoke({
    messages: [{ role: "user", content: "Calculate 25 multiplied by 4" }],
    selected_tool_ids: []
  });
  
  const mathMessage = mathResult.messages[mathResult.messages.length - 1];
  console.log(`   Query: "Calculate 25 multiplied by 4"`);
  console.log(`   Retrieved tools: ${mathResult.selected_tool_ids.join(', ')}`);
  console.log(`   Response: ${mathMessage.content}\n`);
  
  // Test 3: Complex query requiring multiple tools
  console.log("6. Testing complex query...");
  const complexResult = await agent.invoke({
    messages: [{ 
      role: "user", 
      content: "What's the weather in London and what time is it there?" 
    }],
    selected_tool_ids: []
  });
  
  const complexMessage = complexResult.messages[complexResult.messages.length - 1];
  console.log(`   Query: "What's the weather in London and what time is it there?"`);
  console.log(`   Retrieved tools: ${complexResult.selected_tool_ids.join(', ')}`);
  console.log(`   Response: ${complexMessage.content}\n`);
  
  console.log("=== Example Complete ===");
}

// Run the example
if (import.meta.url === `file://${process.argv[1]}`) {
  main().catch(console.error);
}

export { main };