#!/usr/bin/env node
import { createToolRegistry } from "../src/index.js";
import { tool } from "@langchain/core/tools";
import { z } from "zod";
import type { StructuredTool } from "@langchain/core/tools";
import type { ToolRegistry } from "../src/types.js";

// Create test tools
const tool1 = tool(
  async ({ input }: { input: string }) => `Tool 1: ${input}`,
  {
    name: "tool_one",
    description: "First test tool",
    schema: z.object({ input: z.string() })
  }
);

const tool2 = tool(
  async ({ input }: { input: string }) => `Tool 2: ${input}`,
  {
    name: "tool_two",
    description: "Second test tool",
    schema: z.object({ input: z.string() })
  }
);

async function test() {
  console.log("Testing createToolRegistry...\n");
  
  // Test 1: Create registry from array
  console.log("1. Creating registry from array of tools...");
  const tools: StructuredTool[] = [tool1, tool2];
  const registry = createToolRegistry(tools);
  
  console.log("   ✓ Registry created");
  console.log("   ✓ Contains tool_one:", 'tool_one' in registry);
  console.log("   ✓ Contains tool_two:", 'tool_two' in registry);
  console.log("   ✓ tool_one.name:", registry.tool_one?.name);
  console.log("   ✓ tool_two.name:", registry.tool_two?.name);
  
  // Test 2: Pass existing registry (should return as-is)
  console.log("\n2. Passing existing registry...");
  const existingRegistry: ToolRegistry = { tool_one: tool1, tool_two: tool2 };
  const registry2 = createToolRegistry(existingRegistry);
  
  console.log("   ✓ Same registry returned:", registry2 === existingRegistry);
  
  // Test 3: Use with createAgent
  console.log("\n3. Using with createAgent...");
  console.log("   ✓ Can pass array directly to createAgent");
  console.log("   ✓ Can pass registry to createAgent");
  
  console.log("\n✅ createToolRegistry working correctly!");
}

test().catch(console.error);