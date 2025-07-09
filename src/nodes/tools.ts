import { RunnableConfig } from "@langchain/core/runnables";
import { ToolMessage } from "@langchain/core/messages";
import { isAIMessage } from "@langchain/core/messages";
import { BigToolState, BigToolConfig } from "../types.js";

export async function selectTools(
  state: BigToolState,
  config: RunnableConfig
): Promise<Partial<BigToolState>> {
  if (!state.messages || state.messages.length === 0) {
    return {};
  }
  
  const lastMessage = state.messages[state.messages.length - 1];
  
  // Check if the last message is an AI message with tool calls
  if (!isAIMessage(lastMessage) || !lastMessage.tool_calls?.length) {
    return {};
  }
  
  const bigToolConfig = config as BigToolConfig;
  const toolMessages: ToolMessage[] = [];
  const newToolIds: string[] = [];
  
  // Process each tool call
  for (const toolCall of lastMessage.tool_calls) {
    if (toolCall.name === "retrieve_tools") {
      const { query } = toolCall.args as { query: string };
      
      // Retrieve relevant tool IDs
      const retrievedIds = await bigToolConfig.retrieveFunction(
        query,
        { 
          limit: bigToolConfig.limit, 
          filter: bigToolConfig.filter 
        },
        bigToolConfig.store
      );
      
      // Add retrieved IDs to the list
      newToolIds.push(...retrievedIds);
      
      // Format the tool descriptions for the response
      const toolDescriptions = retrievedIds
        .map(id => {
          const tool = bigToolConfig.toolRegistry[id];
          if (!tool) return null;
          return `- ${tool.name}: ${tool.description}`;
        })
        .filter(Boolean)
        .join("\n");
      
      const content = toolDescriptions
        ? `Found ${retrievedIds.length} relevant tools:\n${toolDescriptions}`
        : "No relevant tools found for your query.";
      
      // Create tool message
      toolMessages.push(
        new ToolMessage({
          content,
          tool_call_id: toolCall.id || "",
          name: "retrieve_tools"
        })
      );
    }
  }
  
  return {
    messages: toolMessages,
    selected_tool_ids: newToolIds
  };
}