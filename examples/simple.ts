import { createAgent } from "../src/index.js";
import { InMemoryStore } from "@langchain/langgraph";
import { HumanMessage } from "@langchain/core/messages";
import type { BaseChatModel } from "@langchain/core/language_models/chat_models";
import type { StructuredTool } from "@langchain/core/tools";
import type { ToolRegistry } from "../src/types.js";

// Simple mock LLM for testing
class MockLLM implements Partial<BaseChatModel> {
  bindTools(tools: StructuredTool[]) {
    console.log(`Mock LLM bound with ${tools.length} tools:`, tools.map(t => t.name));
    return {
      invoke: async (messages: any) => {
        console.log("Mock LLM invoked with messages:", messages.length);
        return {
          content: "Mock response",
          tool_calls: []
        };
      }
    } as any;
  }
}

async function test() {
  console.log("Testing LangGraph BigTool TypeScript Implementation");
  console.log("=".repeat(50));
  
  // Create simple tool registry
  const toolRegistry: ToolRegistry = {
    tool1: { 
      name: "tool1", 
      description: "First tool", 
      invoke: async () => "Result 1" 
    } as any,
    tool2: { 
      name: "tool2", 
      description: "Second tool", 
      invoke: async () => "Result 2" 
    } as any
  };
  
  // Set up store
  const store = new InMemoryStore();
  
  // Create mock LLM
  const llm = new MockLLM() as any;
  
  // Create agent
  console.log("\nCreating agent...");
  const agent = await createAgent({
    llm: llm,
    tools: toolRegistry,
    store,
    options: { limit: 2 }
  });
  
  console.log("Agent created successfully!");
  console.log("\nAgent structure:");
  console.log("- Type:", typeof agent);
  console.log("- Has invoke method:", typeof agent.invoke === 'function');
  console.log("- Has stream method:", typeof agent.stream === 'function');
  
  // Agent is already compiled
  console.log("\nAgent is ready to use!");
  
  // Test state initialization
  const initialState = {
    messages: [new HumanMessage("Test message")],
    selected_tool_ids: []
  };
  
  console.log("\nInitial state:", initialState);
  console.log("\nTest completed successfully!");
}

test().catch(console.error);