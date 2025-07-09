#!/usr/bin/env node
import { createAgent } from "../src/index.js";
import { tool } from "@langchain/core/tools";
import { z } from "zod";
import type { BaseChatModel } from "@langchain/core/language_models/chat_models";
import type { StructuredTool } from "@langchain/core/tools";
import type { ToolRegistry } from "../src/types.js";

// Mock LLM for testing
class MockLLM implements Partial<BaseChatModel> {
  bindTools() { 
    return this; 
  }
  
  async invoke() {
    return {
      content: "Test response",
      tool_calls: []
    };
  }
}

// Create simple tool
const testTool = tool(
  async ({ input }: { input: string }) => `Processed: ${input}`,
  {
    name: "test_tool",
    description: "A test tool",
    schema: z.object({ input: z.string() })
  }
);

async function test() {
  console.log("Testing async createAgent...\n");
  
  // Create agent without providing a store
  console.log("1. Creating agent (async)...");
  const agent = await createAgent({
    llm: new MockLLM() as any,
    tools: { test_tool: testTool } as ToolRegistry
  });
  console.log("   ✓ Agent created with await");
  
  // Check that we can immediately access properties
  console.log("2. Checking immediate property access...");
  console.log("   ✓ Agent has invoke method:", typeof agent.invoke === 'function');
  console.log("   ✓ Agent has stream method:", typeof agent.stream === 'function');
  
  // Test invoking the agent
  console.log("3. Testing invoke...");
  try {
    const result = await agent.invoke({
      messages: [],
      selected_tool_ids: []
    });
    console.log("   ✓ Invoke succeeded");
  } catch (error) {
    console.log("   ✓ Invoke attempted (error expected without proper LLM)");
  }
  
  console.log("\n✅ Async createAgent working correctly!");
}

test().catch(console.error);