import { describe, it, expect, beforeEach } from "@jest/globals";
import { createAgent } from "../src/index.js";
import { InMemoryStore } from "@langchain/langgraph";
import { tool } from "@langchain/core/tools";
import { ChatOpenAI } from "@langchain/openai";
import { z } from "zod";
import { HumanMessage, AIMessage } from "@langchain/core/messages";
import { ToolRegistry } from "../src/types.js";

// Mock OpenAI to avoid API calls
jest.mock("@langchain/openai", () => ({
  ChatOpenAI: jest.fn().mockImplementation(() => ({
    bindTools: jest.fn().mockImplementation((tools) => ({
      invoke: jest.fn().mockResolvedValue(
        new AIMessage({
          content: "Test response",
          tool_calls: []
        })
      )
    }))
  }))
}));

describe("Default Tools Feature", () => {
  let store: InMemoryStore;
  let mockLLM: any;
  let addTool: any;
  let multiplyTool: any;
  let subtractTool: any;

  beforeEach(() => {
    store = new InMemoryStore();
    
    // Create a mock LLM that tracks tool bindings
    const boundTools: any[] = [];
    mockLLM = {
      bindTools: jest.fn().mockImplementation((tools) => {
        boundTools.length = 0;
        boundTools.push(...tools);
        return {
          invoke: jest.fn().mockResolvedValue(
            new AIMessage({
              content: "Test response",
              tool_calls: []
            })
          ),
          _boundTools: boundTools
        };
      })
    };

    // Create test tools
    addTool = tool(
      async ({ a, b }) => `${a} + ${b} = ${a + b}`,
      {
        name: "add",
        description: "Add two numbers",
        schema: z.object({
          a: z.number(),
          b: z.number()
        })
      }
    );

    multiplyTool = tool(
      async ({ a, b }) => `${a} * ${b} = ${a * b}`,
      {
        name: "multiply",
        description: "Multiply two numbers",
        schema: z.object({
          a: z.number(),
          b: z.number()
        })
      }
    );

    subtractTool = tool(
      async ({ a, b }) => `${a} - ${b} = ${a - b}`,
      {
        name: "subtract",
        description: "Subtract two numbers",
        schema: z.object({
          a: z.number(),
          b: z.number()
        })
      }
    );
  });

  it("should include default tools in agent without retrieval", async () => {
    const defaultTools = { add: addTool };
    const registryTools = { multiply: multiplyTool, subtract: subtractTool };

    // Index registry tools
    await store.put(["tools"], "multiply", { 
      tool_id: "multiply",
      description: "Multiply two numbers"
    });
    
    await store.put(["tools"], "subtract", { 
      tool_id: "subtract",
      description: "Subtract two numbers"
    });

    const agent = await createAgent({
      llm: mockLLM,
      tools: registryTools,
      defaultTools,
      store
    });

    // Invoke agent to trigger tool binding
    await agent.invoke({
      messages: [new HumanMessage("Test message")],
      selected_tool_ids: []
    });

    // Verify bindTools was called
    expect(mockLLM.bindTools).toHaveBeenCalled();
    
    // Get the tools that were bound
    const boundModel = mockLLM.bindTools.mock.results[0].value;
    const boundTools = boundModel._boundTools;

    // Check that default tool is included
    const toolNames = boundTools.map((t: any) => t.name);
    expect(toolNames).toContain("retrieve_tools");
    expect(toolNames).toContain("add");
    expect(toolNames).not.toContain("multiply");
    expect(toolNames).not.toContain("subtract");
  });

  it("should include both default and selected tools", async () => {
    const defaultTools = { add: addTool };
    const registryTools = { multiply: multiplyTool, subtract: subtractTool };

    const agent = await createAgent({
      llm: mockLLM,
      tools: registryTools,
      defaultTools,
      store
    });

    // Invoke with selected tools
    await agent.invoke({
      messages: [new HumanMessage("Test message")],
      selected_tool_ids: ["multiply"]
    });

    // Get the tools that were bound
    const boundModel = mockLLM.bindTools.mock.results[0].value;
    const boundTools = boundModel._boundTools;
    const toolNames = boundTools.map((t: any) => t.name);

    // Check that both default and selected tools are included
    expect(toolNames).toContain("retrieve_tools");
    expect(toolNames).toContain("add");
    expect(toolNames).toContain("multiply");
    expect(toolNames).not.toContain("subtract");
  });

  it("should work without default tools (backward compatibility)", async () => {
    const registryTools = { multiply: multiplyTool, subtract: subtractTool };

    const agent = await createAgent({
      llm: mockLLM,
      tools: registryTools,
      store
    });

    await agent.invoke({
      messages: [new HumanMessage("Test message")],
      selected_tool_ids: []
    });

    const boundModel = mockLLM.bindTools.mock.results[0].value;
    const boundTools = boundModel._boundTools;
    const toolNames = boundTools.map((t: any) => t.name);

    // Only retrieve_tools should be bound
    expect(toolNames).toContain("retrieve_tools");
    expect(toolNames).not.toContain("add");
    expect(toolNames).not.toContain("multiply");
    expect(toolNames).not.toContain("subtract");
  });

  it("should handle multiple default tools", async () => {
    const defaultTools = { 
      add: addTool,
      subtract: subtractTool 
    };
    const registryTools = { multiply: multiplyTool };

    const agent = await createAgent({
      llm: mockLLM,
      tools: registryTools,
      defaultTools,
      store
    });

    await agent.invoke({
      messages: [new HumanMessage("Test message")],
      selected_tool_ids: []
    });

    const boundModel = mockLLM.bindTools.mock.results[0].value;
    const boundTools = boundModel._boundTools;
    const toolNames = boundTools.map((t: any) => t.name);

    // All default tools should be included
    expect(toolNames).toContain("retrieve_tools");
    expect(toolNames).toContain("add");
    expect(toolNames).toContain("subtract");
    expect(toolNames).not.toContain("multiply");
  });

  it("should handle array input for default tools", async () => {
    const defaultTools = [addTool, subtractTool];
    const registryTools = { multiply: multiplyTool };

    const agent = await createAgent({
      llm: mockLLM,
      tools: registryTools,
      defaultTools,
      store
    });

    await agent.invoke({
      messages: [new HumanMessage("Test message")],
      selected_tool_ids: []
    });

    const boundModel = mockLLM.bindTools.mock.results[0].value;
    const boundTools = boundModel._boundTools;
    const toolNames = boundTools.map((t: any) => t.name);

    // All default tools should be included
    expect(toolNames).toContain("retrieve_tools");
    expect(toolNames).toContain("add");
    expect(toolNames).toContain("subtract");
    expect(toolNames).not.toContain("multiply");
  });
});