import { Annotation, MessagesAnnotation } from "@langchain/langgraph";
import { StructuredTool, Tool, DynamicStructuredTool } from "@langchain/core/tools";
import { BaseStore } from "@langchain/langgraph";
import { RunnableConfig } from "@langchain/core/runnables";
import { BaseChatModel } from "@langchain/core/language_models/chat_models";

// Custom reducer for tool IDs - adds new IDs without duplicates
export const addNew = (left: string[], right: string[]) => {
  const existing = new Set(left);
  const newIds = right.filter(id => !existing.has(id));
  return [...left, ...newIds];
};

// Extend MessagesAnnotation with selected tools
export const BigToolAnnotation = Annotation.Root({
  // Use the standard MessagesAnnotation for proper message handling
  ...MessagesAnnotation.spec,
  selected_tool_ids: Annotation<string[]>({
    reducer: addNew,
    default: () => []
  })
});

export type BigToolState = typeof BigToolAnnotation.State;

export type ToolRegistry = Record<string, StructuredTool | Tool | DynamicStructuredTool>;

export type ToolInput = ToolRegistry | (StructuredTool | Tool | DynamicStructuredTool)[];

export type RetrieveToolsFunction = (
  query: string,
  config?: {
    limit?: number;
    filter?: Record<string, any>;
  },
  store?: BaseStore
) => Promise<string[]>;

export interface CreateAgentOptions {
  limit?: number;
  filter?: Record<string, any>;
  namespace_prefix?: string[];
  retrieve_tools_function?: RetrieveToolsFunction;
}

export interface CreateAgentInput {
  llm: BaseChatModel;
  tools: ToolInput;
  prompt?: string;
  options?: CreateAgentOptions;
  store?: BaseStore;
}

export interface BigToolConfig extends RunnableConfig {
  store?: BaseStore;
  model: BaseChatModel;
  toolRegistry: ToolRegistry;
  retrieveFunction: RetrieveToolsFunction;
  limit: number;
  filter?: Record<string, any>;
  retrieveTool: any; // Simplified to avoid type issues
  systemPrompt?: string;
}