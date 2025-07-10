import { BaseStore } from "@langchain/langgraph";
import { RetrieveToolsFunction } from "../types.js";

export function getDefaultRetrievalTool(
  namespace_prefix: string[] = ["tools"],
  limit: number = 2,
  filter?: Record<string, any>,
  store?: BaseStore
): RetrieveToolsFunction {
  return async (
    query: string,
    config?: {
      limit?: number;
      filter?: Record<string, any>;
    },
  ): Promise<string[]> => {
    if (!store) {
      throw new Error("Store is required for default retrieval function");
    }
    
    const searchLimit = config?.limit || limit;
    const searchFilter = { ...filter, ...config?.filter };
    
    try {
      // Search for similar documents in the store
      const results = await store.search(
        namespace_prefix,
        {
          query,
          limit: searchLimit,
          filter: searchFilter
        }
      );
      
      // Extract tool IDs from results
      const toolIds: string[] = [];
      
      for (const result of results) {
        if (result.value && typeof result.value === 'object' && 'tool_id' in result.value) {
          toolIds.push(result.value.tool_id as string);
        } else if (result.key) {
          // Use the key as tool ID if no specific tool_id in value
          toolIds.push(result.key);
        }
      }
      
      return toolIds;
    } catch (error) {
      // If search is not implemented, return empty array
        if (error instanceof Error) {
          console.warn("Error message:", error.message);
          console.warn("Error stack:", error.stack);
        } else {
        console.warn("Unknown error:", error);
      }

      return [];
    }
  };
}