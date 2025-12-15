import { StateGraph, START, END } from "@langchain/langgraph";
import { RunnableConfig } from "@langchain/core/runnables";
import { AIMessage, ToolMessage } from "@langchain/core/messages";
import { StructuredToolInterface } from "@langchain/core/tools";
import {
  BigToolAnnotation,
  BigToolState,
  CreateAgentInput,
  BigToolConfig,
  WorkflowOptions,
} from "./types.js";
import { callModel } from "./nodes/agent.js";
import { selectTools } from "./nodes/tools.js";
import { shouldContinue } from "./nodes/routing.js";
import { createRetrieveToolsTool } from "./tools/retrieve.js";
import { getDefaultRetrievalTool } from "./utils/retrieval.js";
import { createToolRegistry } from "./utils/registry.js";
import { SCRATCHPAD_UPDATE_SYMBOL } from "./utils/constants.js";

// Marker used to identify scratchpad update requests from tools
const SCRATCHPAD_MARKER = "__SCRATCHPAD_UPDATE__";

export async function createAgent(
  input: CreateAgentInput,
  workflowOptions: WorkflowOptions = {}
) {
  const { llm, tools, defaultTools, prompt, options = {}, store, checkpointer } = input;
  const toolRegistry = createToolRegistry(tools);
  const defaultToolRegistry = defaultTools
    ? createToolRegistry(defaultTools)
    : undefined;
  const {
    limit = 2,
    filter,
    namespace_prefix = ["tools"],
    retrieve_tools_function = getDefaultRetrievalTool(
      namespace_prefix,
      limit,
      filter,
      store
    ),
  } = options;

  // If store has indexTools method, call it to index the tools
  if (
    store &&
    "indexTools" in store &&
    typeof store.indexTools === "function"
  ) {
    await store.indexTools(toolRegistry);
  }

  // Store prompt to be used in agent node
  const systemPrompt = prompt;

  // Create retrieve_tools tool
  const retrieveTool = createRetrieveToolsTool(
    retrieve_tools_function,
    toolRegistry
  );


  const toolNode = async (state: BigToolState, config: RunnableConfig) => {
    // Get only the selected tools from the registry
    const selectedTools = state.selected_tool_ids
      .map((id) => toolRegistry[id])
      .filter(Boolean);

    const toolsToUse = selectedTools.length > 0 ? selectedTools : [];

    // Get default tools if they exist
    const defaultToolsList = defaultToolRegistry
      ? Object.values(defaultToolRegistry)
      : [];

    // All available tools (retrieve + defaults + selected)
    const allTools: StructuredToolInterface[] = [
      retrieveTool,
      ...defaultToolsList,
      ...toolsToUse,
    ];

    // Create a map for quick tool lookup
    const toolMap = new Map<string, StructuredToolInterface>();
    for (const tool of allTools) {
      toolMap.set(tool.name, tool);
    }

    // Get tool calls from the last message
    const lastMessage = state.messages[state.messages.length - 1];
    const toolCalls =
      lastMessage && "_getType" in lastMessage && lastMessage._getType() === "ai"
        ? (lastMessage as AIMessage).tool_calls || []
        : [];

    if (toolCalls.length === 0) {
      return { messages: [] };
    }

    // SEQUENTIAL EXECUTION: Execute tools one by one, updating scratchpad between each
    let currentScratchpad: Record<string, any> = { ...(state.scratchpad || {}) };
    const resultMessages: ToolMessage[] = [];

    for (const toolCall of toolCalls) {
      const tool = toolMap.get(toolCall.name);

      if (!tool) {
        // Tool not found - create error message
        resultMessages.push(
          new ToolMessage({
            content: `Tool "${toolCall.name}" not found`,
            tool_call_id: toolCall.id || "",
            name: toolCall.name,
          })
        );
        continue;
      }

      try {
        // Execute tool with CURRENT scratchpad state
        const toolConfig: RunnableConfig = {
          ...config,
          configurable: {
            ...config.configurable,
            scratchpad: currentScratchpad, // Each tool sees up-to-date scratchpad
          },
        };

        const rawResult = await tool.invoke(toolCall.args, toolConfig);

        // Parse result to check for scratchpad updates
        let parsedResult: any = rawResult;
        if (typeof rawResult === "string") {
          try {
            parsedResult = JSON.parse(rawResult);
          } catch {
            // Not JSON, use as-is
          }
        }

        // Check if this is a scratchpad update marker
        const isScratchpadUpdate =
          parsedResult &&
          typeof parsedResult === "object" &&
          (parsedResult[SCRATCHPAD_UPDATE_SYMBOL] === true ||
            parsedResult[SCRATCHPAD_MARKER] === true);

        if (isScratchpadUpdate && parsedResult.updates) {
          // IMMEDIATELY update scratchpad so next tool sees it
          currentScratchpad = { ...currentScratchpad, ...parsedResult.updates };

          // Create confirmation message for LLM
          resultMessages.push(
            new ToolMessage({
              content: `Scratchpad updated: ${Object.keys(parsedResult.updates).join(", ")}`,
              tool_call_id: toolCall.id || "",
              name: toolCall.name,
            })
          );
        } else {
          // Normal tool result
          const content =
            typeof rawResult === "string" ? rawResult : JSON.stringify(rawResult);
          resultMessages.push(
            new ToolMessage({
              content,
              tool_call_id: toolCall.id || "",
              name: toolCall.name,
            })
          );
        }
      } catch (error: any) {
        // Handle tool errors based on workflowOptions
        const errorMessage = error?.message || String(error);
        if (workflowOptions.handleToolErrors) {
          resultMessages.push(
            new ToolMessage({
              content: `Error: ${errorMessage}`,
              tool_call_id: toolCall.id || "",
              name: toolCall.name,
            })
          );
        } else {
          throw error;
        }
      }
    }

    // Return updated messages and scratchpad
    // LangGraph's reducer will merge scratchpad updates into state
    return {
      messages: resultMessages,
      scratchpad: currentScratchpad,
    };
  };

  // Create a custom call model that includes config
  const callModelWithConfig = async (
    state: BigToolState,
    config: RunnableConfig
  ) => {
    // Enhance config with our custom properties
    const enhancedConfig: BigToolConfig = {
      ...config,
      model: llm,
      toolRegistry,
      defaultToolRegistry,
      retrieveFunction: retrieve_tools_function,
      retrieveTool,
      limit,
      filter,
      systemPrompt,
    };

    return callModel(state, enhancedConfig);
  };

  // Create a custom select tools that includes config
  const selectToolsWithConfig = async (
    state: BigToolState,
    config: RunnableConfig
  ) => {
    // Enhance config with our custom properties
    const enhancedConfig: BigToolConfig = {
      ...config,
      model: llm,
      toolRegistry,
      defaultToolRegistry,
      retrieveFunction: retrieve_tools_function,
      retrieveTool,
      limit,
      filter,
      systemPrompt,
    };

    return selectTools(state, enhancedConfig);
  };

  // Build graph
  const workflow = new StateGraph(BigToolAnnotation)
    .addNode("agent", callModelWithConfig)
    .addNode("select_tools", selectToolsWithConfig)
    .addNode("tools", toolNode)
    .addEdge(START, "agent")
    .addConditionalEdges("agent", shouldContinue, {
      select_tools: "select_tools",
      tools: "tools",
      [END]: END,
    })
    .addEdge("select_tools", "agent")
    .addEdge("tools", "agent");

  return workflow.compile({
    ...(store && { store }),
    ...(checkpointer && { checkpointer }),
  });
}
