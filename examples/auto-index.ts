#!/usr/bin/env node
import { createAgent, createToolRegistry } from "../src/index.js";
import { tool } from "@langchain/core/tools";
import { z } from "zod";
import { BaseStore } from "@langchain/langgraph";
import type { BaseChatModel } from "@langchain/core/language_models/chat_models";
import type { StructuredTool } from "@langchain/core/tools";
import type { ToolRegistry } from "../src/types.js";

// Mock LLM for testing
class MockLLM implements Partial<BaseChatModel> {
  bindTools() { return this as any; }
  async invoke() {
    return {
      content: "Test response",
      tool_calls: []
    };
  }
}

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

// Mock Store to track indexTools calls
class MockStore extends BaseStore {
  indexToolsCalled = false;
  indexedTools: string[] | null = null;
  
  async indexTools(toolRegistry: ToolRegistry) {
    this.indexToolsCalled = true;
    this.indexedTools = Object.keys(toolRegistry);
    console.log("   ✓ indexTools called with:", this.indexedTools);
  }
  
  // Implement required BaseStore methods
  async get(namespace: string[], key: string) {
    return null;
  }
  
  async put(namespace: string[], key: string, value: any) {
    return;
  }
  
  async delete(namespace: string[], key: string) {
    return;
  }
  
  async list(namespace: string[]) {
    return [];
  }
  
  async search(namespace: string[], query: any) {
    return [];
  }
}

async function test() {
  console.log("Testing automatic tool indexing...\n");
  
  // Create tools
  const tools: StructuredTool[] = [tool1, tool2];
  const toolRegistry = createToolRegistry(tools);
  
  // Create mock store
  console.log("1. Creating mock store...");
  const store = new MockStore();
  console.log("   ✓ Store created");
  console.log("   ✓ indexTools not called yet:", !store.indexToolsCalled);
  
  // Create agent - should automatically call indexTools
  console.log("\n2. Creating agent with store...");
  const agent = await createAgent({
    llm: new MockLLM() as any,
    tools: toolRegistry,
    store
  });
  
  console.log("   ✓ Agent created");
  console.log("   ✓ indexTools was called:", store.indexToolsCalled);
  console.log("   ✓ Correct tools indexed:", 
    store.indexedTools && 
    store.indexedTools.includes('tool_one') && 
    store.indexedTools.includes('tool_two')
  );
  
  console.log("\n✅ Automatic tool indexing working correctly!");
}

test().catch(console.error);