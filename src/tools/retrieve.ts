import { DynamicStructuredTool } from "@langchain/core/tools";
import { z } from "zod";
import { RetrieveToolsFunction, ToolRegistry } from "../types.js";
import { formatToolDescriptions } from "../utils/formatting.js";

// Define the input schema for the retrieve tools function
const retrieveToolsSchema = z.object({
  query: z.string().describe("Query to search for relevant tools")
});

export const createRetrieveToolsTool = (
  retrieveFn: RetrieveToolsFunction,
  toolRegistry: ToolRegistry
) => {
  const tool = new DynamicStructuredTool({
    name: "retrieve_tools",
    description: "Retrieve tools based on a query. Use this to search for and discover available tools.",
    schema: retrieveToolsSchema as any,
    func: async (input: { query: string }) => {
      const toolIds = await retrieveFn(input.query);
      
      // Format the response with tool names and descriptions
      const toolDescriptions = formatToolDescriptions(toolIds, toolRegistry);
      
      if (toolDescriptions !== "No tools found.") {
        return `Found ${toolIds.length} relevant tools:\n${toolDescriptions}`;
      } else {
        return "No relevant tools found for your query.";
      }
    }
  });
  
  return tool;
};