import { describe, it, expect, beforeEach } from "@jest/globals";
import { createAgent } from "../src/graph.js";
import { InMemoryStore } from "@langchain/langgraph";
import { tool } from "@langchain/core/tools";
import { z } from "zod";
import { HumanMessage } from "@langchain/core/messages";
import { FakeChatModel } from "@langchain/core/utils/testing";
import { addNew } from "../src/types.js";

describe("LangGraph BigTool", () => {
  let store: InMemoryStore;
  let toolRegistry: any;
  let fakeLLM: FakeChatModel;

  beforeEach(() => {
    store = new InMemoryStore();
    
    // Create test tools
    toolRegistry = {
      add: tool(
        async ({ a, b }) => `${a} + ${b} = ${a + b}`,
        {
          name: "add",
          description: "Add two numbers",
          schema: z.object({
            a: z.number(),
            b: z.number()
          })
        }
      ),
      multiply: tool(
        async ({ a, b }) => `${a} * ${b} = ${a * b}`,
        {
          name: "multiply", 
          description: "Multiply two numbers",
          schema: z.object({
            a: z.number(),
            b: z.number()
          })
        }
      )
    };

    // Create fake LLM for testing
    fakeLLM = new FakeChatModel({
      responses: []
    });
  });

  describe("State Reducer", () => {
    it("should add new tool IDs without duplicates", () => {
      const left = ["tool1", "tool2"];
      const right = ["tool2", "tool3", "tool4"];
      const result = addNew(left, right);
      
      expect(result).toEqual(["tool1", "tool2", "tool3", "tool4"]);
    });

    it("should preserve order when adding new IDs", () => {
      const left = ["a", "b"];
      const right = ["c", "d"];
      const result = addNew(left, right);
      
      expect(result).toEqual(["a", "b", "c", "d"]);
    });
  });

  describe("Agent Creation", () => {
    it("should create agent with default options", async () => {
      const agent = await createAgent({
        llm: fakeLLM,
        tools: toolRegistry
      });
      expect(agent).toBeDefined();
      expect(agent.invoke).toBeDefined();
      expect(agent.stream).toBeDefined();
    });

    it("should create agent with custom options", async () => {
      const agent = await createAgent({
        llm: fakeLLM,
        tools: toolRegistry,
        options: {
          limit: 5,
          filter: { category: "math" },
          namespace_prefix: ["custom", "tools"]
        }
      });
      expect(agent).toBeDefined();
    });

    it("should create agent with custom retrieval function", async () => {
      const customRetriever = async (query: string) => {
        return query.includes("add") ? ["add"] : ["multiply"];
      };

      const agent = await createAgent({
        llm: fakeLLM,
        tools: toolRegistry,
        options: {
          retrieve_tools_function: customRetriever
        }
      });
      expect(agent).toBeDefined();
    });
  });

  describe("Tool Registry", () => {
    it("should accept structured tools", async () => {
      const agent = await createAgent({
        llm: fakeLLM,
        tools: toolRegistry,
        store
      });
      expect(agent).toBeDefined();
      expect(agent.invoke).toBeDefined();
    });
  });
});