import { createAgent } from "../src/index.js";
import { InMemoryStore } from "@langchain/langgraph";
import { HumanMessage, AIMessage } from "@langchain/core/messages";
import { DynamicStructuredTool } from "@langchain/core/tools";
import { z } from "zod";
import type { StructuredTool } from "@langchain/core/tools";
import type { BaseChatModel } from "@langchain/core/language_models/chat_models";
import type { RetrieveToolsFunction } from "../src/types.js";

// Create tools as array
const tools: StructuredTool[] = [
  new DynamicStructuredTool({
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
  }),
  new DynamicStructuredTool({
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
  }),
  new DynamicStructuredTool({
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
  })
];

// Mock LLM
class MockLLM implements Partial<BaseChatModel> {
  private callCount = 0;
  
  bindTools(tools: StructuredTool[]) {
    const toolNames = tools.map(t => t.name);
    console.log(`\nLLM bound with ${tools.length} tools:`, toolNames);
    
    return {
      invoke: async (messages: any) => {
        this.callCount++;
        console.log(`\nLLM Call #${this.callCount}:`);
        console.log("Messages received:", messages.length);
        console.log("Last message:", messages[messages.length - 1].content);
        
        // First call: use retrieve_tools
        if (this.callCount === 1) {
          console.log("-> Calling retrieve_tools");
          return new AIMessage({
            content: "I'll search for math tools to help with subtraction.",
            tool_calls: [{
              id: "call_1",
              name: "retrieve_tools",
              args: { query: "subtract difference" }
            }]
          });
        }
        
        // Second call: use retrieved tool
        if (this.callCount === 2 && toolNames.includes("subtract")) {
          console.log("-> Using 'subtract' tool");
          return new AIMessage({
            content: "Now I'll subtract those numbers for you.",
            tool_calls: [{
              id: "call_2",
              name: "subtract",
              args: { a: 10, b: 4 }
            }]
          });
        }
        
        // Final response
        console.log("-> Final response");
        return new AIMessage({
          content: "The result of 10 - 4 is 6."
        });
      }
    } as any;
  }
}

// Custom retrieval function
const customRetriever: RetrieveToolsFunction = async (query: string) => {
  console.log(`\nRetrieval function called with query: "${query}"`);
  
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
  console.log("Testing LangGraph BigTool with Array Input");
  console.log("=".repeat(60));
  
  const store = new InMemoryStore();
  const llm = new MockLLM() as any;
  
  // Create agent with array of tools
  console.log("\nCreating agent with array of tools...");
  const agent = await createAgent({
    llm: llm,
    tools: tools,
    options: {
      limit: 2,
      retrieve_tools_function: customRetriever
    }
  });
  
  console.log("Agent created successfully!");
  console.log("✅ Array input format works correctly!");
  
  // Test the agent
  console.log("\n" + "=".repeat(60));
  console.log("Testing: 'Calculate 10 - 4'");
  console.log("=".repeat(60));
  
  try {
    const result = await agent.invoke({
      messages: [new HumanMessage("Calculate 10 - 4")],
      selected_tool_ids: []
    });
    
    console.log("\n" + "=".repeat(60));
    console.log("Final Result:");
    console.log("=".repeat(60));
    console.log("Selected tools:", result.selected_tool_ids);
    console.log("\nLast message:", result.messages[result.messages.length - 1].content);
    
  } catch (error) {
    console.error("\nError during execution:", error);
  }
}

test().catch(console.error);