import { BaseStore, Item, Operation, OperationResults } from "@langchain/langgraph";
import { HNSWLib, HNSWLibArgs } from "@langchain/community/vectorstores/hnswlib";
import { Document } from "@langchain/core/documents";
import { Embeddings } from "@langchain/core/embeddings";
import type { ToolRegistry } from "../types.js";

interface ToolData {
  tool_id: string;
  name: string;
  description: string;
  embedding?: number[];
}


interface SearchResult extends Item {
  score: number;
}

/**
 * HNSWLib-based store for high-performance vector search.
 * 
 * @example
 * ```typescript
 * import { HTTPEmbeddings, HNSWLibStore } from "langgraph-bigtool";
 * 
 * const embeddings = new HTTPEmbeddings({ serviceUrl: 'http://localhost:8001' });
 * const store = new HNSWLibStore(embeddings, {
 *   space: 'cosine',  // optional, defaults to 'cosine'
 *   numDimensions: 384  // optional, auto-detected from first embedding
 * });
 * 
 * // Tools are automatically indexed when creating agent
 * const agent = await createAgent({ llm, tools, store });
 * ```
 */
export class HNSWLibStore extends BaseStore {
  private embeddings: Embeddings;
  private vectorStore: HNSWLib;
  private toolMetadata = new Map<string, ToolData>();
  
  constructor(embeddings: Embeddings, hnswConfig?: HNSWLibArgs) {
    super();
    this.embeddings = embeddings;
    
    // Initialize HNSWLib with provided config or defaults
    this.vectorStore = new HNSWLib(this.embeddings, {
      space: 'cosine',
      ...hnswConfig
    });
  }
  
  /**
   * Index tools for semantic search using HNSWLib
   */
  async indexTools(toolRegistry: ToolRegistry): Promise<void> {
    // Prepare documents
    const entries = Object.entries(toolRegistry);
    const documents = entries.map(([toolId, tool]) => {
      const doc = new Document({
        pageContent: `${tool.name} ${tool.description || ""}`,
        metadata: {
          tool_id: toolId,
          name: tool.name,
          description: tool.description || ""
        }
      });
      
      // Store metadata for later retrieval
      this.toolMetadata.set(toolId, {
        tool_id: toolId,
        name: tool.name,
        description: tool.description || ""
      });
      
      return doc;
    });
    
    // Add documents to existing vector store
    await this.vectorStore.addDocuments(documents);
  }
  
  /**
   * Search for tools using HNSWLib vector search
   */
  async search(
    namespacePrefix: string[],
    options?: {
      filter?: Record<string, any>;
      limit?: number;
      offset?: number;
      query?: string;
    }
  ): Promise<SearchResult[]> {
    const limit = options?.limit || 10;
    const query = options?.query || "";
    
    if (!query) {
      // No query, return all tools up to limit
      const results: SearchResult[] = [];
      let count = 0;
      for (const [toolId, metadata] of this.toolMetadata) {
        if (count >= limit) break;
        results.push({
          value: metadata,
          key: toolId,
          namespace: namespacePrefix,
          score: 1,
          createdAt: new Date(),
          updatedAt: new Date()
        });
        count++;
      }
      return results;
    }
    
    // Perform vector similarity search
    const searchResults = await this.vectorStore.similaritySearchWithScore(query, limit);
    
    return searchResults.map(([doc, score]) => ({
      value: this.toolMetadata.get(doc.metadata.tool_id) || {
        tool_id: doc.metadata.tool_id,
        name: doc.metadata.name,
        description: doc.metadata.description
      },
      key: doc.metadata.tool_id,
      namespace: namespacePrefix,
      score,
      createdAt: new Date(),
      updatedAt: new Date()
    }));
  }
  
  /**
   * Required BaseStore methods
   */
  
  async get(namespace: string[], key: string): Promise<Item | null> {
    const metadata = this.toolMetadata.get(key);
    if (!metadata) return null;
    
    return {
      value: metadata,
      key,
      namespace,
      createdAt: new Date(),
      updatedAt: new Date()
    };
  }
  
  async put(_namespace: string[], key: string, value: ToolData): Promise<void> {
    this.toolMetadata.set(key, value);
    
    // Add to vector store if it has a description
    if (value.description) {
      const doc = new Document({
        pageContent: `${value.name} ${value.description}`,
        metadata: {
          tool_id: key,
          name: value.name,
          description: value.description
        }
      });
      await this.vectorStore.addDocuments([doc]);
    }
  }
  
  async delete(_namespace: string[], key: string): Promise<void> {
    this.toolMetadata.delete(key);
    // Note: HNSWLib doesn't support deletion, would need to rebuild index
  }
  
  async list(_namespace: string[]): Promise<Item[]> {
    const results: Item[] = [];
    for (const [key, value] of this.toolMetadata) {
      results.push({
        value,
        key,
        namespace: _namespace,
        createdAt: new Date(),
        updatedAt: new Date()
      });
    }
    return results;
  }
  
  async batch<Op extends Operation[]>(
    operations: Op
  ): Promise<OperationResults<Op>> {
    const results: any[] = [];
    
    for (const op of operations) {
      if ('value' in op && 'namespace' in op && 'key' in op) {
        // Put operation
        await this.put(op.namespace, op.key, op.value as ToolData);
        results.push(undefined);
      } else if ('namespace' in op && 'key' in op) {
        // Get operation
        const value = await this.get(op.namespace, op.key);
        results.push(value);
      } else {
        // Other operation types
        results.push(null);
      }
    }
    
    return results as OperationResults<Op>;
  }
  
  async *start(): AsyncGenerator<void> {
    // No initialization needed
    yield;
  }
  
}