import { createAgent } from "../src/index.js";
import { InMemoryStore } from "@langchain/langgraph";
import { HumanMessage, AIMessage } from "@langchain/core/messages";
import { DynamicStructuredTool } from "@langchain/core/tools";
import { z } from "zod";
import type { BaseStore } from "@langchain/langgraph";
import type { BaseChatModel } from "@langchain/core/language_models/chat_models";
import type { StructuredTool } from "@langchain/core/tools";
import type { ToolRegistry, RetrieveToolsFunction } from "../src/types.js";
import type { BaseMessage } from "@langchain/core/messages";

// Create proper LangGraph tools
const addTool = new DynamicStructuredTool({
  name: "add",
  description: "Add two numbers together",
  schema: z.object({
    a: z.number().describe("First number"),
    b: z.number().describe("Second number")
  }),
  func: async ({ a, b }: { a: number; b: number }) => {
    console.log(`Tool 'add' called with a=${a}, b=${b}`);
    return `${a} + ${b} = ${a + b}`;
  }
});

const multiplyTool = new DynamicStructuredTool({
  name: "multiply",
  description: "Multiply two numbers",
  schema: z.object({
    a: z.number().describe("First number"),
    b: z.number().describe("Second number")
  }),
  func: async ({ a, b }: { a: number; b: number }) => {
    console.log(`Tool 'multiply' called with a=${a}, b=${b}`);
    return `${a} × ${b} = ${a * b}`;
  }
});

const subtractTool = new DynamicStructuredTool({
  name: "subtract",
  description: "Subtract one number from another",
  schema: z.object({
    a: z.number().describe("First number to subtract from"),
    b: z.number().describe("Second number to subtract")
  }),
  func: async ({ a, b }: { a: number; b: number }) => {
    console.log(`Tool 'subtract' called with a=${a}, b=${b}`);
    return `${a} - ${b} = ${a - b}`;
  }
});

// Mock LLM that simulates tool retrieval
class MockRetrievalLLM implements Partial<BaseChatModel> {
  private callCount: number = 0;
  
  bindTools(tools: StructuredTool[]) {
    const toolNames = tools.map(t => t.name);
    console.log(`\nLLM bound with ${tools.length} tools:`, toolNames);
    
    return {
      invoke: async (messages: BaseMessage[]) => {
        this.callCount++;
        console.log(`\nLLM Call #${this.callCount}:`);
        console.log("Last message:", messages[messages.length - 1].content);
        
        // First call: use retrieve_tools
        if (this.callCount === 1) {
          console.log("-> Calling retrieve_tools");
          return new AIMessage({
            content: "I'll search for math tools to help with your calculation.",
            tool_calls: [{
              id: "call_1",
              name: "retrieve_tools",
              args: { query: "math calculation add" }
            }]
          });
        }
        
        // Second call: after tools are retrieved, use them
        if (this.callCount === 2 && toolNames.includes("add")) {
          console.log("-> Using 'add' tool");
          return new AIMessage({
            content: "Now I'll add those numbers for you.",
            tool_calls: [{
              id: "call_2",
              name: "add",
              args: { a: 5, b: 3 }
            }]
          });
        }
        
        // Final response
        console.log("-> Final response");
        return new AIMessage({
          content: "The result of 5 + 3 is 8. The calculation has been completed successfully."
        });
      }
    } as any;
  }
}

// Create tool registry
const toolRegistry: ToolRegistry = {
  add: addTool,
  multiply: multiplyTool,
  subtract: subtractTool
};

// Custom retrieval function
const customRetriever: RetrieveToolsFunction = async (query: string) => {
  console.log(`\nRetrieval function called with query: "${query}"`);
  
  // Simple keyword matching
  const queryLower = query.toLowerCase();
  const results: string[] = [];
  
  if (queryLower.includes("add") || queryLower.includes("sum")) {
    results.push("add");
  }
  if (queryLower.includes("multiply") || queryLower.includes("product")) {
    results.push("multiply");
  }
  if (queryLower.includes("subtract") || queryLower.includes("difference")) {
    results.push("subtract");
  }
  
  console.log(`-> Retrieved tools: ${results.join(", ")}`);
  return results;
};

async function test() {
  console.log("Testing LangGraph BigTool with Proper LangGraph Tools");
  console.log("=".repeat(60));
  
  // Set up store
  const store = new InMemoryStore();
  
  // Create mock LLM
  const llm = new MockRetrievalLLM() as any;
  
  // Create agent with custom retrieval
  console.log("\nCreating agent with dynamic tool selection...");
  const agent = await createAgent({
    llm: llm,
    tools: toolRegistry,
    store,
    options: {
      limit: 2,
      retrieve_tools_function: customRetriever
    }
  });
  
  console.log("Agent compiled successfully!");
  
  // Test the agent
  console.log("\n" + "=".repeat(60));
  console.log("Testing: 'Calculate 5 + 3'");
  console.log("=".repeat(60));
  
  try {
    const result = await agent.invoke({
      messages: [new HumanMessage("Calculate 5 + 3")],
      selected_tool_ids: []
    });
    
    console.log("\n" + "=".repeat(60));
    console.log("Final Result:");
    console.log("=".repeat(60));
    console.log("Selected tools:", result.selected_tool_ids);
    console.log("\nConversation history:");
    result.messages.forEach((msg: BaseMessage, i: number) => {
      console.log(`\n${i + 1}. ${msg.constructor.name}:`);
      console.log(`   Content: ${msg.content}`);
      if ('tool_calls' in msg && msg.tool_calls) {
        console.log(`   Tool calls:`, msg.tool_calls);
      }
    });
    
  } catch (error) {
    console.error("\nError during execution:", error);
  }
}

test().catch(console.error);