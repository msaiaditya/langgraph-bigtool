import { END } from "@langchain/langgraph";
import { isAIMessage } from "@langchain/core/messages";
import { BigToolState } from "../types.js";

export function shouldContinue(state: Partial<BigToolState>): string {
  if (!state.messages || state.messages.length === 0) {
    return END;
  }
  
  const lastMessage = state.messages[state.messages.length - 1];
  
  // If the last message is not an AI message, end
  if (!isAIMessage(lastMessage)) {
    return END;
  }
  
  // If there are no tool calls, end
  if (!lastMessage.tool_calls || lastMessage.tool_calls.length === 0) {
    return END;
  }
  
  // Check if any tool call is for retrieve_tools
  const hasRetrieveTools = lastMessage.tool_calls.some(
    toolCall => toolCall.name === "retrieve_tools"
  );
  
  if (hasRetrieveTools) {
    return "select_tools";
  }
  
  // Check if we have any selected tools before routing to tools node
  // If no tools are selected and it's not a retrieve_tools call, we should end
  if (!state.selected_tool_ids || state.selected_tool_ids.length === 0) {
    // However, we still route to tools node as it now handles this case
    return "tools";
  }
  
  // Otherwise, execute the tools
  return "tools";
}