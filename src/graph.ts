import { StateGraph, START, END } from "@langchain/langgraph";
import { ToolNode } from "@langchain/langgraph/prebuilt";
import { RunnableConfig } from "@langchain/core/runnables";
import { 
  BigToolAnnotation, 
  BigToolState, 
  CreateAgentInput,
  BigToolConfig 
} from "./types.js";
import { callModel } from "./nodes/agent.js";
import { selectTools } from "./nodes/tools.js";
import { shouldContinue } from "./nodes/routing.js";
import { createRetrieveToolsTool } from "./tools/retrieve.js";
import { getDefaultRetrievalTool } from "./utils/retrieval.js";
import { createToolRegistry } from "./utils/registry.js";

export async function createAgent(input: CreateAgentInput) {
  const { llm, tools, defaultTools, prompt, options = {}, store } = input;
  const toolRegistry = createToolRegistry(tools);
  const defaultToolRegistry = defaultTools ? createToolRegistry(defaultTools) : undefined;
  const {
    limit = 2,
    filter,
    namespace_prefix = ["tools"],
    retrieve_tools_function = getDefaultRetrievalTool(namespace_prefix, limit, filter, store)
  } = options;
  
  // If store has indexTools method, call it to index the tools
  if (store && 'indexTools' in store && typeof store.indexTools === 'function') {
    await store.indexTools(toolRegistry);
  }
  
  // Store prompt to be used in agent node
  const systemPrompt = prompt;
  
  // Create retrieve_tools tool
  const retrieveTool = createRetrieveToolsTool(retrieve_tools_function, toolRegistry);
  
  // Create a custom tool node that only uses selected tools
  const toolNode = async (state: BigToolState, config: RunnableConfig) => {
    // Get only the selected tools from the registry
    const selectedTools = state.selected_tool_ids
      .map(id => toolRegistry[id])
      .filter(Boolean);
    
    // If no tools are selected, we still need to handle tool calls
    // The agent will always have at least retrieve_tools available
    const toolsToUse = selectedTools.length > 0 ? selectedTools : [];
    
    // Get default tools if they exist
    const defaultToolsList = defaultToolRegistry ? Object.values(defaultToolRegistry) : [];
    
    // Always include the retrieve tool and default tools for tool calls
    const allTools = [retrieveTool, ...defaultToolsList, ...toolsToUse];
    
    // Create a ToolNode with the available tools
    const dynamicToolNode = new ToolNode(allTools);
    
    // Execute the tool node
    return dynamicToolNode.invoke(state, config);
  };
  
  // Create a custom call model that includes config
  const callModelWithConfig = async (state: BigToolState, config: RunnableConfig) => {
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
      systemPrompt
    };
    
    return callModel(state, enhancedConfig);
  };
  
  // Create a custom select tools that includes config
  const selectToolsWithConfig = async (state: BigToolState, config: RunnableConfig) => {
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
      systemPrompt
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
      [END]: END
    })
    .addEdge("select_tools", "agent")
    .addEdge("tools", "agent");
  
  // Compile with or without store
  return store ? workflow.compile({ store }) : workflow.compile();
}