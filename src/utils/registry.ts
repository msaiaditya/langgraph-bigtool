import { ToolRegistry, ToolInput } from "../types.js";
import { StructuredTool, Tool, DynamicStructuredTool } from "@langchain/core/tools";
import { Document } from "@langchain/core/documents";

/**
 * Get the unique identifier for a tool
 * @param tool - The tool object
 * @returns The tool's unique identifier (currently the tool name)
 */
export function getToolId(tool: StructuredTool | Tool | DynamicStructuredTool): string {
  return tool.name;
}

/**
 * Create a tool registry from an array of tools or return existing registry
 * 
 * @param tools - Array of tools or existing tool registry
 * @returns Tool registry object with tool names as keys
 * 
 * @example
 * ```typescript
 * const tools = [weatherTool, calculatorTool];
 * const toolRegistry = createToolRegistry(tools);
 * // Result: { get_weather: weatherTool, calculator: calculatorTool }
 * ```
 */
export function createToolRegistry(tools: ToolInput): ToolRegistry {
  if (Array.isArray(tools)) {
    return tools.reduce((registry, tool) => {
      registry[getToolId(tool)] = tool;
      return registry;
    }, {} as ToolRegistry);
  }
  // Already a registry
  return tools;
}

/**
 * Create a Document object from a tool for embedding/storage
 * 
 * @param tool - The tool to convert to a document
 * @returns Document object with tool information
 * 
 * @example
 * ```typescript
 * const tool = weatherTool;
 * const doc = createToolDocument(tool);
 * // Result: Document with pageContent and metadata
 * ```
 */
export function createToolDocument(tool: StructuredTool | Tool | DynamicStructuredTool): Document {
  const toolId = getToolId(tool);
  return new Document({
    pageContent: `${tool.name} ${tool.description || ""}`,
    metadata: {
      tool_id: toolId,
      name: tool.name,
      description: tool.description || ""
    }
  });
}