/**
 * String key used to mark tool outputs that request a scratchpad update.
 * Tools can return an object with this key set to true to indicate
 * that the result should update the scratchpad rather than be returned to the LLM.
 * 
 * NOTE: This MUST be a string, not a Symbol, because Symbols don't serialize to JSON!
 */
export const SCRATCHPAD_UPDATE_SYMBOL = "__SCRATCHPAD_UPDATE__";

