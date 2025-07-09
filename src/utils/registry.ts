import { ToolRegistry, ToolInput } from "../types.js";

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
      registry[tool.name] = tool;
      return registry;
    }, {} as ToolRegistry);
  }
  // Already a registry
  return tools;
}