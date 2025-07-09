#!/usr/bin/env node
import { createAgent } from "../src/index.js";
import { InMemoryStore } from "@langchain/langgraph";
import { tool } from "@langchain/core/tools";
import { ChatOpenAI } from "@langchain/openai";
import { z } from "zod";
import { HumanMessage } from "@langchain/core/messages";

// Create tools from Math functions
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

const powTool = tool(
  async ({ base, exponent }) => {
    const result = Math.pow(base, exponent);
    return `${base} raised to the power of ${exponent} is ${result}`;
  },
  {
    name: "pow",
    description: "Calculate the power of a number (base^exponent)",
    schema: z.object({
      base: z.number().describe("The base number"),
      exponent: z.number().describe("The exponent")
    })
  }
);

const sinTool = tool(
  async ({ angle }) => {
    const result = Math.sin(angle);
    return `The sine of ${angle} radians is ${result}`;
  },
  {
    name: "sin",
    description: "Calculate the sine of an angle in radians",
    schema: z.object({
      angle: z.number().describe("The angle in radians")
    })
  }
);

const cosTool = tool(
  async ({ angle }) => {
    const result = Math.cos(angle);
    return `The cosine of ${angle} radians is ${result}`;
  },
  {
    name: "cos",
    description: "Calculate the cosine of an angle in radians",
    schema: z.object({
      angle: z.number().describe("The angle in radians")
    })
  }
);

const logTool = tool(
  async ({ number }) => {
    const result = Math.log(number);
    return `The natural logarithm of ${number} is ${result}`;
  },
  {
    name: "log",
    description: "Calculate the natural logarithm (ln) of a number",
    schema: z.object({
      number: z.number().describe("The number to calculate natural log of")
    })
  }
);

async function main() {
  // Create tool registry
  const toolRegistry = {
    sqrt: sqrtTool,
    pow: powTool,
    sin: sinTool,
    cos: cosTool,
    log: logTool
  };

  // Set up in-memory store and index tools
  const store = new InMemoryStore();
  
  // Index tools in the store for semantic search
  await store.put(["tools"], "sqrt", { 
    tool_id: "sqrt",
    description: "Calculate the square root of a number",
    keywords: ["square", "root", "sqrt", "radical"]
  });
  
  await store.put(["tools"], "pow", { 
    tool_id: "pow",
    description: "Calculate the power of a number (base^exponent)",
    keywords: ["power", "exponent", "pow", "raise"]
  });
  
  await store.put(["tools"], "sin", { 
    tool_id: "sin",
    description: "Calculate the sine of an angle in radians",
    keywords: ["sine", "sin", "trigonometry", "angle"]
  });
  
  await store.put(["tools"], "cos", { 
    tool_id: "cos",
    description: "Calculate the cosine of an angle in radians",
    keywords: ["cosine", "cos", "trigonometry", "angle"]
  });
  
  await store.put(["tools"], "log", { 
    tool_id: "log",
    description: "Calculate the natural logarithm (ln) of a number",
    keywords: ["logarithm", "log", "ln", "natural"]
  });

  // Create agent
  const llm = new ChatOpenAI({ 
    model: "gpt-4o-mini",
    temperature: 0
  });
  
  const agent = await createAgent({
    llm,
    tools: toolRegistry,
    store,  // Use the manually configured store
    options: { 
      limit: 2  // Only retrieve 2 tools at a time
    }
  });

  // Example 1: Calculate square root
  console.log("Example 1: Calculate square root of 16");
  console.log("-".repeat(50));
  
  const result1 = await agent.invoke({
    messages: [new HumanMessage("Calculate the square root of 16")],
    selected_tool_ids: []
  });
  
  console.log("Final message:", result1.messages[result1.messages.length - 1].content);
  console.log();

  // Example 2: Calculate power
  console.log("Example 2: Calculate 2 raised to the power of 8");
  console.log("-".repeat(50));
  
  const result2 = await agent.invoke({
    messages: [new HumanMessage("What is 2 raised to the power of 8?")],
    selected_tool_ids: []
  });
  
  console.log("Final message:", result2.messages[result2.messages.length - 1].content);
  console.log();

  // Example 3: Multiple calculations
  console.log("Example 3: Complex calculation");
  console.log("-".repeat(50));
  
  const result3 = await agent.invoke({
    messages: [new HumanMessage("First find the sine of pi/2, then calculate the natural logarithm of 10")],
    selected_tool_ids: []
  });
  
  console.log("Final message:", result3.messages[result3.messages.length - 1].content);
  console.log();

  // Show selected tools
  console.log("Selected tools for last query:", result3.selected_tool_ids);
}

// Run the example
if (import.meta.url === `file://${process.argv[1]}`) {
  main().catch(console.error);
}