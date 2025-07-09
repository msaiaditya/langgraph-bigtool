import { createAgent } from "../src/index.js";
import { InMemoryStore } from "@langchain/langgraph";
import { HumanMessage, AIMessage } from "@langchain/core/messages";
import { DynamicStructuredTool } from "@langchain/core/tools";
import { z } from "zod";
import type { BaseChatModel } from "@langchain/core/language_models/chat_models";
import type { StructuredTool } from "@langchain/core/tools";
import type { ToolRegistry, RetrieveToolsFunction } from "../src/types.js";

// Create tools
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

// Mock LLM that checks for system prompt
class MockLLMWithPromptCheck implements Partial<BaseChatModel> {
  private callCount = 0;
  systemPromptReceived = false;
  
  bindTools(tools: StructuredTool[]) {
    const toolNames = tools.map(t => t.name);
    console.log(`\nLLM bound with ${tools.length} tools:`, toolNames);
    
    return {
      invoke: async (messages: any) => {
        this.callCount++;
        console.log(`\nLLM Call #${this.callCount}:`);
        console.log("Total messages received:", messages.length);
        
        // Check for system message
        if (messages.length > 0 && messages[0]._getType() === "system") {
          this.systemPromptReceived = true;
          console.log("✅ System prompt detected:", messages[0].content);
        }
        
        console.log("Last message:", messages[messages.length - 1].content);
        
        // First call: use retrieve_tools
        if (this.callCount === 1) {
          console.log("-> Calling retrieve_tools");
          return new AIMessage({
            content: "As a helpful math assistant, I'll search for the appropriate tools.",
            tool_calls: [{
              id: "call_1",
              name: "retrieve_tools",
              args: { query: "add sum calculation" }
            }]
          });
        }
        
        // Second call: use retrieved tool
        if (this.callCount === 2 && toolNames.includes("add")) {
          console.log("-> Using 'add' tool");
          return new AIMessage({
            content: "I'll add those numbers for you.",
            tool_calls: [{
              id: "call_2",
              name: "add",
              args: { a: 7, b: 5 }
            }]
          });
        }
        
        // Final response
        console.log("-> Final response");
        return new AIMessage({
          content: "The sum of 7 and 5 is 12. Is there anything else I can help you calculate?"
        });
      }
    } as any;
  }
}

// Tool registry
const toolRegistry: ToolRegistry = {
  add: addTool,
  multiply: multiplyTool
};

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
  
  console.log(`-> Retrieved tools: ${results.join(", ")}`);
  return results;
};

async function test() {
  console.log("Testing LangGraph BigTool with System Prompt");
  console.log("=".repeat(60));
  
  const store = new InMemoryStore();
  const llm = new MockLLMWithPromptCheck() as any;
  
  // Create agent with system prompt
  const systemPrompt = "You are a helpful math assistant. Always be polite and explain your steps clearly.";
  
  console.log("\nCreating agent with system prompt...");
  console.log("Prompt:", systemPrompt);
  
  const agent = await createAgent({
    llm: llm,
    tools: toolRegistry,
    prompt: systemPrompt,
    options: {
      limit: 2,
      retrieve_tools_function: customRetriever
    }
  });
  
  console.log("\nAgent created successfully!");
  
  // Test the agent
  console.log("\n" + "=".repeat(60));
  console.log("Testing: 'Add 7 and 5'");
  console.log("=".repeat(60));
  
  try {
    const result = await agent.invoke({
      messages: [new HumanMessage("Add 7 and 5")],
      selected_tool_ids: []
    });
    
    console.log("\n" + "=".repeat(60));
    console.log("Final Result:");
    console.log("=".repeat(60));
    console.log("System prompt was received:", llm.systemPromptReceived);
    console.log("Selected tools:", result.selected_tool_ids);
    console.log("\nFinal AI response:", result.messages[result.messages.length - 1].content);
    
    if (llm.systemPromptReceived) {
      console.log("\n✅ System prompt functionality works correctly!");
    } else {
      console.log("\n❌ System prompt was not detected!");
    }
    
  } catch (error) {
    console.error("\nError during execution:", error);
  }
}

test().catch(console.error);