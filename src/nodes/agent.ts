import { RunnableConfig } from "@langchain/core/runnables";
import { SystemMessage } from "@langchain/core/messages";
import { BigToolState, BigToolConfig } from "../types.js";

export async function callModel(
  state: BigToolState,
  config: RunnableConfig
): Promise<Partial<BigToolState>> {
  const { messages, selected_tool_ids = [] } = state;
  const bigToolConfig = config as BigToolConfig;
  
  // Get selected tools from registry
  const selectedTools = selected_tool_ids
    .map(id => bigToolConfig.toolRegistry[id])
    .filter(Boolean);
  
  // Get default tools if they exist
  const defaultTools = bigToolConfig.defaultToolRegistry 
    ? Object.values(bigToolConfig.defaultToolRegistry)
    : [];
  
  // Always include retrieve_tools, default tools, and selected tools
  const tools = [bigToolConfig.retrieveTool, ...defaultTools, ...selectedTools];
  
  // Bind tools to model
  if (!bigToolConfig.model.bindTools) {
    throw new Error("Model does not support tool binding");
  }
  
  const modelWithTools = bigToolConfig.model.bindTools(tools);
  
  // Prepare messages with optional system prompt
  let messagesToSend = messages;
  if (bigToolConfig.systemPrompt && messages.length > 0) {
    // Check if first message is already a system message
    const hasSystemMessage = messages[0]._getType() === "system";
    if (!hasSystemMessage) {
      const systemMessage = new SystemMessage(bigToolConfig.systemPrompt);
      messagesToSend = [systemMessage, ...messages];
    }
  }
  
  // Invoke the model with messages
  const response = await modelWithTools.invoke(messagesToSend, config);
  
  return { messages: [response] };
}