#!/usr/bin/env node
import { createAgent } from "../src/index.js";
import { InMemoryStore } from "@langchain/langgraph";
import { tool } from "@langchain/core/tools";
import { ChatOpenAI } from "@langchain/openai";
import { z } from "zod";
import { HumanMessage } from "@langchain/core/messages";

// Create default tools that are always available
const calculatorTool = tool(
  async ({ operation, a, b }) => {
    let result: number;
    switch (operation) {
      case "add":
        result = a + b;
        break;
      case "subtract":
        result = a - b;
        break;
      case "multiply":
        result = a * b;
        break;
      case "divide":
        if (b === 0) return "Error: Division by zero";
        result = a / b;
        break;
      default:
        return "Error: Unknown operation";
    }
    return `${a} ${operation} ${b} = ${result}`;
  },
  {
    name: "calculator",
    description: "Basic calculator for add, subtract, multiply, divide operations",
    schema: z.object({
      operation: z.enum(["add", "subtract", "multiply", "divide"]),
      a: z.number().describe("First operand"),
      b: z.number().describe("Second operand")
    })
  }
);

const getCurrentTimeTool = tool(
  async () => {
    const now = new Date();
    return `Current time: ${now.toISOString()}`;
  },
  {
    name: "getCurrentTime",
    description: "Get the current date and time",
    schema: z.object({})
  }
);

// Create advanced tools that need to be discovered
const sqrtTool = tool(
  async ({ number }) => {
    const result = Math.sqrt(number);
    return `The square root of ${number} is ${result}`;
  },
  {
    name: "sqrt",
    description: "Calculate the square root of a number",
    schema: z.object({
      number: z.number().describe("The number to calculate square root of")
    })
  }
);

const factorialTool = tool(
  async ({ n }) => {
    if (n < 0) return "Error: Factorial not defined for negative numbers";
    if (n === 0 || n === 1) return `${n}! = 1`;
    
    let result = 1;
    for (let i = 2; i <= n; i++) {
      result *= i;
    }
    return `${n}! = ${result}`;
  },
  {
    name: "factorial",
    description: "Calculate the factorial of a non-negative integer",
    schema: z.object({
      n: z.number().int().nonnegative().describe("The number to calculate factorial of")
    })
  }
);

const fibonacciTool = tool(
  async ({ n }) => {
    if (n < 0) return "Error: Fibonacci not defined for negative numbers";
    if (n === 0) return "F(0) = 0";
    if (n === 1) return "F(1) = 1";
    
    let a = 0, b = 1;
    for (let i = 2; i <= n; i++) {
      const temp = a + b;
      a = b;
      b = temp;
    }
    return `F(${n}) = ${b}`;
  },
  {
    name: "fibonacci",
    description: "Calculate the nth Fibonacci number",
    schema: z.object({
      n: z.number().int().nonnegative().describe("The position in Fibonacci sequence")
    })
  }
);

async function main() {
  // Default tools - always available without retrieval
  const defaultTools = {
    calculator: calculatorTool,
    getCurrentTime: getCurrentTimeTool
  };

  // Advanced tools - need to be discovered via semantic search
  const advancedTools = {
    sqrt: sqrtTool,
    factorial: factorialTool,
    fibonacci: fibonacciTool
  };

  // Set up store and index advanced tools only
  const store = new InMemoryStore();
  
  // Only index the advanced tools (not default tools)
  await store.put(["tools"], "sqrt", { 
    tool_id: "sqrt",
    description: "Calculate the square root of a number",
    keywords: ["square", "root", "sqrt", "radical", "math", "advanced"]
  });
  
  await store.put(["tools"], "factorial", { 
    tool_id: "factorial",
    description: "Calculate the factorial of a non-negative integer",
    keywords: ["factorial", "permutation", "combination", "math", "advanced"]
  });
  
  await store.put(["tools"], "fibonacci", { 
    tool_id: "fibonacci",
    description: "Calculate the nth Fibonacci number",
    keywords: ["fibonacci", "sequence", "series", "math", "advanced"]
  });

  // Create agent with default tools
  const llm = new ChatOpenAI({ 
    model: "gpt-4o-mini",
    temperature: 0
  });
  
  const agent = await createAgent({
    llm,
    tools: advancedTools,  // Tools that need to be discovered
    defaultTools,          // Tools that are always available
    store,
    options: { 
      limit: 2  // Only retrieve 2 tools at a time
    }
  });

  // Example 1: Use default calculator tool (no retrieval needed)
  console.log("Example 1: Using default calculator tool");
  console.log("-".repeat(50));
  
  const result1 = await agent.invoke({
    messages: [new HumanMessage("What is 15 + 27?")],
    selected_tool_ids: []
  });
  
  console.log("Final message:", result1.messages[result1.messages.length - 1].content);
  console.log("Selected tools:", result1.selected_tool_ids);
  console.log();

  // Example 2: Use default time tool (no retrieval needed)
  console.log("Example 2: Using default time tool");
  console.log("-".repeat(50));
  
  const result2 = await agent.invoke({
    messages: [new HumanMessage("What time is it right now?")],
    selected_tool_ids: []
  });
  
  console.log("Final message:", result2.messages[result2.messages.length - 1].content);
  console.log("Selected tools:", result2.selected_tool_ids);
  console.log();

  // Example 3: Use advanced tool (requires retrieval)
  console.log("Example 3: Using advanced tool (requires retrieval)");
  console.log("-".repeat(50));
  
  const result3 = await agent.invoke({
    messages: [new HumanMessage("Calculate the square root of 144")],
    selected_tool_ids: []
  });
  
  console.log("Final message:", result3.messages[result3.messages.length - 1].content);
  console.log("Selected tools:", result3.selected_tool_ids);
  console.log();

  // Example 4: Complex calculation using both default and advanced tools
  console.log("Example 4: Using both default and advanced tools");
  console.log("-".repeat(50));
  
  const result4 = await agent.invoke({
    messages: [new HumanMessage("First calculate 5 factorial, then add 25 to the result")],
    selected_tool_ids: []
  });
  
  console.log("Final message:", result4.messages[result4.messages.length - 1].content);
  console.log("Selected tools:", result4.selected_tool_ids);
  console.log();

  // Example 5: Show that default tools work even with unrelated queries
  console.log("Example 5: Default tools always available");
  console.log("-".repeat(50));
  
  const result5 = await agent.invoke({
    messages: [new HumanMessage("I need to find the 10th Fibonacci number and also tell me what time it is")],
    selected_tool_ids: []
  });
  
  console.log("Final message:", result5.messages[result5.messages.length - 1].content);
  console.log("Selected tools:", result5.selected_tool_ids);
}

// Run the example
if (import.meta.url === `file://${process.argv[1]}`) {
  main().catch(console.error);
}