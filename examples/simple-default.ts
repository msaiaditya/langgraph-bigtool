#!/usr/bin/env node
/**
 * Simple example showing usage without a store
 * All tools are available to the agent immediately
 */

import { createAgent, createToolRegistry } from "../src/index.js";
import { tool } from "@langchain/core/tools";
import { ChatOpenAI } from "@langchain/openai";
import { HumanMessage } from "@langchain/core/messages";
import { z } from "zod";

// Create some example tools
const weatherTool = tool(
  async ({ location }) => `Weather in ${location}: Sunny, 72°F`,
  {
    name: "get_weather",
    description: "Get the current weather for a location",
    schema: z.object({ location: z.string() })
  }
);

const calculatorTool = tool(
  async ({ a, b, op }) => {
    switch(op) {
      case "add": return `${a} + ${b} = ${a + b}`;
      case "subtract": return `${a} - ${b} = ${a - b}`;
      case "multiply": return `${a} * ${b} = ${a * b}`;
      case "divide": return `${a} / ${b} = ${a / b}`;
      default: return "Unknown operation";
    }
  },
  {
    name: "calculator",
    description: "Perform arithmetic operations: add, subtract, multiply, divide",
    schema: z.object({
      a: z.number(),
      b: z.number(),
      op: z.enum(["add", "subtract", "multiply", "divide"])
    })
  }
);

const translateTool = tool(
  async ({ text, to }) => {
    const translations: Record<string, Record<string, string>> = {
      "Hello": { "spanish": "Hola", "french": "Bonjour" },
      "Goodbye": { "spanish": "Adiós", "french": "Au revoir" }
    };
    return translations[text]?.[to] || "Translation not available";
  },
  {
    name: "translate",
    description: "Translate common phrases to Spanish or French",
    schema: z.object({
      text: z.string(),
      to: z.enum(["spanish", "french"])
    })
  }
);

async function main() {
  console.log("=== Simple Example Without Store ===\n");
  
  const llm = new ChatOpenAI({ 
    model: "gpt-4o-mini",
    temperature: 0
  });
  
  // Create tools array
  const tools = [weatherTool, calculatorTool, translateTool];
  
  // Option 1: Pass tools array directly (createAgent handles conversion)
  console.log("Creating agent without a store (using array)...");
  const agent = await createAgent({
    llm,
    tools,  // Can pass array directly
    // No store - all tools are available to the agent
    options: {
      limit: 2  // Retrieve up to 2 tools at a time
    }
  });
  
  // Option 2: Create registry explicitly (useful when you need the registry elsewhere)
  // const toolRegistry = createToolRegistry(tools);
  // const agent = await createAgent({ llm, tools: toolRegistry });
  console.log("✓ Agent created with all tools available!\n");
  
  // Test queries
  const queries = [
    "What's the weather in Paris?",
    "Calculate 25 + 17",
    "How do you say Hello in Spanish?"
  ];
  
  for (const query of queries) {
    console.log(`Query: "${query}"`);
    
    const result = await agent.invoke({
      messages: [new HumanMessage(query)],
      selected_tool_ids: []
    });
    
    const response = result.messages[result.messages.length - 1].content;
    console.log("Response:", response);
    console.log("Tools used:", result.selected_tool_ids);
    console.log();
  }
  
  console.log("Note: Without a store:");
  console.log("- All tools are available to the agent immediately");
  console.log("- No tool discovery needed for small tool sets");
  console.log("- Works on all platforms without dependencies");
  console.log("- For large tool sets, consider using a store for better performance");
}

if (import.meta.url === `file://${process.argv[1]}`) {
  main().catch(console.error);
}