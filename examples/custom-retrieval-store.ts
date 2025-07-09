#!/usr/bin/env node
/**
 * Example of implementing custom retrieval with stores for the langgraph-bigtool library.
 * Shows both keyword-based and embeddings-based approaches.
 */

import { createAgent } from "../src/index.js";
import { InMemoryStore } from "@langchain/langgraph";
import { tool } from "@langchain/core/tools";
import { ChatOpenAI } from "@langchain/openai";
import { HumanMessage } from "@langchain/core/messages";
import { z } from "zod";
import type { RetrieveToolsFunction, ToolRegistry } from "../src/types.js";

// Example tools
const weatherTool = tool(
  async ({ location }) => {
    return `The weather in ${location} is sunny and 72°F`;
  },
  {
    name: "get_weather",
    description: "Get the current weather for a location",
    schema: z.object({
      location: z.string().describe("The location to get weather for")
    })
  }
);

const calculatorTool = tool(
  async ({ expression }) => {
    try {
      const result = eval(expression);
      return `Result: ${result}`;
    } catch (error) {
      return `Error: Invalid expression`;
    }
  },
  {
    name: "calculator",
    description: "Perform mathematical calculations",
    schema: z.object({
      expression: z.string().describe("Mathematical expression to evaluate")
    })
  }
);

const timezoneTool = tool(
  async ({ city }) => {
    const timezones: Record<string, string> = {
      "New York": "3:45 PM EST",
      "London": "8:45 PM GMT",
      "Tokyo": "5:45 AM JST",
      "Sydney": "7:45 AM AEDT"
    };
    return `The current time in ${city} is ${timezones[city] || "unknown"}`;
  },
  {
    name: "get_timezone",
    description: "Get the current time in a specific city",
    schema: z.object({
      city: z.string().describe("The city to get the time for")
    })
  }
);

// Create tool registry
const toolRegistry: ToolRegistry = {
  get_weather: weatherTool,
  calculator: calculatorTool,
  get_timezone: timezoneTool
};

/**
 * Example 1: Simple keyword-based retrieval
 */
async function keywordRetrievalExample() {
  console.log("=== Example 1: Keyword-based Tool Retrieval ===\n");
  
  // Create store and index tools with keywords
  const store = new InMemoryStore();
  
  // Index tools with keywords for better search
  await store.put(["tools"], "get_weather", {
    tool_id: "get_weather",
    description: "Get weather forecast temperature climate conditions",
    keywords: ["weather", "temperature", "forecast", "climate", "sunny", "rain"]
  });
  
  await store.put(["tools"], "calculator", {
    tool_id: "calculator",
    description: "Calculate math arithmetic operations add subtract multiply divide",
    keywords: ["math", "calculate", "arithmetic", "add", "subtract", "multiply"]
  });
  
  await store.put(["tools"], "get_timezone", {
    tool_id: "get_timezone",
    description: "Get time timezone clock hours city location",
    keywords: ["time", "timezone", "clock", "hours", "GMT", "EST", "PST"]
  });
  
  // Create custom retrieval function
  const keywordRetrieval: RetrieveToolsFunction = async (query, config, store) => {
    if (!store) throw new Error("Store required");
    
    const queryTerms = query.toLowerCase().split(/\s+/);
    const toolScores = new Map<string, number>();
    
    // Get all tools
    const items = await store.list(["tools"]);
    
    // Score each tool based on keyword matches
    for (const item of items) {
      let score = 0;
      const itemText = `${item.value.description} ${item.value.keywords?.join(" ") || ""}`.toLowerCase();
      
      for (const term of queryTerms) {
        if (itemText.includes(term)) {
          score += 1;
        }
      }
      
      if (score > 0) {
        toolScores.set(item.value.tool_id, score);
      }
    }
    
    // Sort by score and return top tools
    const sortedTools = Array.from(toolScores.entries())
      .sort((a, b) => b[1] - a[1])
      .slice(0, config?.limit || 2)
      .map(([toolId]) => toolId);
    
    return sortedTools;
  };
  
  // Create agent with custom retrieval
  const llm = new ChatOpenAI({ 
    model: "gpt-4o-mini",
    temperature: 0
  });
  
  const agent = await createAgent({
    llm,
    tools: toolRegistry,
    store,  // Use the manually configured store
    options: {
      retrieve_tools_function: keywordRetrieval,
      limit: 2
    }
  });
  
  // Test queries
  console.log("Query: What's the weather like in Paris?");
  const result1 = await agent.invoke({
    messages: [new HumanMessage("What's the weather like in Paris?")],
    selected_tool_ids: []
  });
  
  console.log("Response:", result1.messages[result1.messages.length - 1].content);
  console.log("Tools found:", result1.selected_tool_ids);
  console.log();
}

/**
 * Example 2: Embeddings-based retrieval (with cosine similarity)
 */
async function embeddingsRetrievalExample() {
  console.log("=== Example 2: Embeddings-based Tool Retrieval ===\n");
  
  // Simple cosine similarity function
  function cosineSimilarity(a: number[], b: number[]): number {
    let dotProduct = 0;
    let normA = 0;
    let normB = 0;
    
    for (let i = 0; i < a.length; i++) {
      dotProduct += a[i] * b[i];
      normA += a[i] * a[i];
      normB += b[i] * b[i];
    }
    
    return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
  }
  
  // Create store
  const store = new InMemoryStore();
  
  // For this example, we'll use mock embeddings
  // In production, use real embeddings from OpenAI or HuggingFace
  const mockEmbeddings: Record<string, number[]> = {
    get_weather: [0.8, 0.2, 0.1, 0.5], // Weather-related embedding
    calculator: [0.1, 0.9, 0.3, 0.2],  // Math-related embedding
    get_timezone: [0.3, 0.1, 0.8, 0.6] // Time-related embedding
  };
  
  // Index tools with embeddings
  for (const [toolId, embedding] of Object.entries(mockEmbeddings)) {
    await store.put(["tools"], toolId, {
      tool_id: toolId,
      embedding: embedding
    });
  }
  
  // Create embeddings-based retrieval function
  const embeddingsRetrieval: RetrieveToolsFunction = async (query, config, store) => {
    if (!store) throw new Error("Store required");
    
    // Mock query embedding (in production, use real embeddings)
    const queryEmbedding = query.includes("weather") ? [0.7, 0.3, 0.2, 0.4] :
                          query.includes("math") || query.includes("calculate") ? [0.2, 0.8, 0.4, 0.3] :
                          query.includes("time") ? [0.4, 0.2, 0.7, 0.5] :
                          [0.5, 0.5, 0.5, 0.5]; // Default
    
    // Get all tools and calculate similarities
    const items = await store.list(["tools"]);
    const similarities: Array<{ toolId: string; similarity: number }> = [];
    
    for (const item of items) {
      if (item.value.embedding) {
        const similarity = cosineSimilarity(queryEmbedding, item.value.embedding);
        similarities.push({ 
          toolId: item.value.tool_id, 
          similarity 
        });
      }
    }
    
    // Sort by similarity and return top tools
    const topTools = similarities
      .sort((a, b) => b.similarity - a.similarity)
      .slice(0, config?.limit || 2)
      .map(item => item.toolId);
    
    return topTools;
  };
  
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
      retrieve_tools_function: embeddingsRetrieval,
      limit: 2
    }
  });
  
  // Test semantic queries
  console.log("Query: I need to do some arithmetic");
  const result = await agent.invoke({
    messages: [new HumanMessage("I need to do some arithmetic: 25 * 4")],
    selected_tool_ids: []
  });
  
  console.log("Response:", result.messages[result.messages.length - 1].content);
  console.log("Tools found:", result.selected_tool_ids);
}

/**
 * Example 3: Using with real embeddings (requires additional setup)
 */
async function realEmbeddingsExample() {
  console.log("\n=== Example 3: Real Embeddings Setup ===\n");
  
  console.log("To use real embeddings:");
  console.log("1. Install: npm install @langchain/community @huggingface/transformers");
  console.log("2. Import: import { HuggingFaceTransformersEmbeddings } from '@langchain/community/embeddings/huggingface_transformers';");
  console.log("3. Create embeddings: const embeddings = new HuggingFaceTransformersEmbeddings({ model: 'Xenova/all-MiniLM-L6-v2' });");
  console.log("4. Use embeddings.embedQuery() and embeddings.embedDocuments() to generate real embeddings");
  console.log("\nThis provides semantic search without network calls!");
}

// Run examples
async function main() {
  try {
    await keywordRetrievalExample();
    await embeddingsRetrievalExample();
    await realEmbeddingsExample();
  } catch (error) {
    console.error("Error:", error);
  }
}

if (import.meta.url === `file://${process.argv[1]}`) {
  main().catch(console.error);
}