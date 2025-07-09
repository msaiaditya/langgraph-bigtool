import { ToolRegistry } from "../types.js";

export function formatToolDescriptions(
  toolIds: string[],
  toolRegistry: ToolRegistry
): string {
  const descriptions = toolIds
    .map(id => {
      const tool = toolRegistry[id];
      if (!tool) return null;
      return `- ${tool.name}: ${tool.description}`;
    })
    .filter(Boolean)
    .join("\n");
  
  return descriptions || "No tools found.";
}