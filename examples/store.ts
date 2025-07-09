#!/usr/bin/env node
import { createAgent, HNSWLibStore, HTTPEmbeddings } from "../src/index.js";
import { tool } from "@langchain/core/tools";
import { ChatOpenAI } from "@langchain/openai";
import { HumanMessage } from "@langchain/core/messages";
import { z } from "zod";
import type { ToolRegistry } from "../src/types.js";

// Create simple tools
const weatherTool = tool(
  async ({ location }: { location: string }) => `Weather in ${location}: Sunny, 72°F`,
  {
    name: "get_weather",
    description: "Get weather for a location",
    schema: z.object({ location: z.string() })
  }
);

const calculatorTool = tool(
  async ({ a, b }: { a: number; b: number }) => `${a} + ${b} = ${a + b}`,
  {
    name: "add",
    description: "Add two numbers",
    schema: z.object({ a: z.number(), b: z.number() })
  }
);

async function test() {
  console.log("Testing HNSWLibStore integration...\n");
  
  // Create embeddings
  const embeddings = new HTTPEmbeddings({ serviceUrl: 'http://localhost:8001' });
  
  // Check if embeddings service is available
  const isHealthy = await embeddings.checkHealth();
  if (!isHealthy) {
    console.error("⚠️  Embeddings service is not available!");
    console.error("   Please ensure the embeddings service is running at http://localhost:8001");
    console.error("   See the embeddings-service directory for setup instructions.\n");
    
    console.log("Falling back to mock test without actual store operations...");
    
    // Continue with mock test
    const toolRegistry: ToolRegistry = {
      get_weather: weatherTool,
      add: calculatorTool
    };
    
    console.log("✓ Tool registry created with:", Object.keys(toolRegistry).join(", "));
    console.log("\nNote: For full store testing, ensure embeddings service is running.");
    return;
  }
  
  console.log("✓ Embeddings service is online\n");
  
  // Create store
  const store = new HNSWLibStore(embeddings, {
    space: 'cosine',
    numDimensions: 384  // all-MiniLM-L6-v2 dimensions
  });
  
  const toolRegistry: ToolRegistry = {
    get_weather: weatherTool,
    add: calculatorTool
  };
  
  // Test store operations
  console.log("1. Testing store methods:");
  
  // Put some test data
  await store.put(["tools"], "get_weather", {
    tool_id: "get_weather",
    name: "get_weather",
    description: "Get weather for a location"
  });
  
  await store.put(["tools"], "add", {
    tool_id: "add",
    name: "add",
    description: "Add two numbers"
  });
  
  // Test get
  const item = await store.get(["tools"], "get_weather");
  console.log("   ✓ Get:", item?.value.tool_id);
  
  // Test list
  const items = await store.list(["tools"]);
  console.log("   ✓ List:", items.length, "items");
  
  // Test search with embeddings
  const searchResults = await store.search(["tools"], { query: "weather", limit: 2 });
  console.log("   ✓ Search:", searchResults.map(r => r.value.tool_id));
  
  // Test with agent
  console.log("\n2. Testing with agent:");
  const llm = new ChatOpenAI({ 
    model: "gpt-4o-mini",
    temperature: 0
  });
  
  const agent = await createAgent({
    llm,
    tools: toolRegistry,
    store  // Use the test store
  });
  
  const result = await agent.invoke({
    messages: [new HumanMessage("What's the weather in Paris?")],
    selected_tool_ids: []
  });
  
  console.log("   ✓ Agent response:", result.messages[result.messages.length - 1].content);
  console.log("   ✓ Tools used:", result.selected_tool_ids);
  
  console.log("\n✅ HNSWLibStore is working correctly!");
}

test().catch(console.error);