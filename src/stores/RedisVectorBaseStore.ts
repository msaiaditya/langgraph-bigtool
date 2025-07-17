import { BaseStore, Operation, OperationResults } from "@langchain/langgraph";
import { RedisVectorStore } from "@langchain/redis";
import { Document } from "@langchain/core/documents";
import { Embeddings } from "@langchain/core/embeddings";
import { createClient } from "redis";
import { createHash } from "crypto";
import type { ToolRegistry } from "../types.js";
import type { Item, SearchItem } from "@langchain/langgraph-checkpoint";
import { createToolDocument } from "../utils/registry.js";

interface ToolData {
  tool_id: string;
  name: string;
  description: string;
  content_hash: string;
  indexed_at: number;
}

interface RedisStoreConfig {
  redisUrl: string;
  embeddings: Embeddings;
  indexName?: string;
  ttlSeconds?: number;
  verbose?: boolean;
}

/**
 * Redis-based vector store for semantic tool search with persistent storage.
 * Implements intelligent caching to avoid redundant embedding generation.
 * 
 * @example
 * ```typescript
 * import { OpenAIEmbeddings } from "@langchain/openai";
 * import { RedisVectorBaseStore } from "langgraph-bigtool";
 * 
 * const embeddings = new OpenAIEmbeddings({ model: 'text-embedding-3-small' });
 * const store = new RedisVectorBaseStore({
 *   redisUrl: 'redis://localhost:6379',
 *   embeddings,
 *   ttlSeconds: 7 * 24 * 60 * 60 // 7 days
 * });
 * 
 * await store.connect();
 * // Tools are automatically indexed when creating agent
 * const agent = await createAgent({ llm, tools, store });
 * ```
 */
export class RedisVectorBaseStore extends BaseStore {
  private redisClient!: ReturnType<typeof createClient>;
  private vectorStore!: RedisVectorStore;
  private embeddings: Embeddings;
  private indexName: string;
  private ttlSeconds: number;
  private verbose: boolean;
  private redisUrl: string;
  private isConnected = false;
  
  constructor(config: RedisStoreConfig) {
    super();
    this.redisUrl = config.redisUrl;
    this.embeddings = config.embeddings;
    this.indexName = config.indexName || "bigtool-tools";
    this.ttlSeconds = config.ttlSeconds || 7 * 24 * 60 * 60; // 7 days
    this.verbose = config.verbose || false;
  }
  
  /**
   * Connect to Redis and initialize the vector store
   */
  async connect(): Promise<void> {
    if (this.isConnected) return;
    
    this.redisClient = createClient({ url: this.redisUrl });
    await this.redisClient.connect();
    
    this.vectorStore = new RedisVectorStore(this.embeddings, {
      redisClient: this.redisClient as any, // Type mismatch between redis and @langchain/redis
      indexName: this.indexName
    });
    
    this.isConnected = true;
    
    if (this.verbose) {
      console.info(`[RedisVectorBaseStore] Connected to Redis at ${this.redisUrl}`);
    }
  }
  
  /**
   * Disconnect from Redis
   */
  async disconnect(): Promise<void> {
    if (!this.isConnected) return;
    
    await this.redisClient.quit();
    this.isConnected = false;
    
    if (this.verbose) {
      console.info(`[RedisVectorBaseStore] Disconnected from Redis`);
    }
  }
  
  /**
   * Generate SHA256 hash of content
   */
  private generateHash(content: string): string {
    return createHash('sha256').update(content).digest('hex');
  }
  
  /**
   * Execute a function with performance logging
   */
  private async withPerformanceLogging<T>(
    operation: string,
    fn: () => Promise<T>,
    getDetails?: (result: T) => string
  ): Promise<T> {
    if (!this.verbose) {
      return fn();
    }
    
    const startTime = performance.now();
    const result = await fn();
    const endTime = performance.now();
    const duration = endTime - startTime;
    
    const details = getDetails ? getDetails(result) : '';
    console.info(
      `[RedisVectorBaseStore] ${operation} completed in ${duration.toFixed(2)}ms${details ? ' - ' + details : ''}`
    );
    
    return result;
  }
  
  /**
   * Index tools with intelligent caching and batch operations
   */
  async indexTools(toolRegistry: ToolRegistry): Promise<void> {
    await this.ensureConnected();
    
    const toolIds = Object.keys(toolRegistry);
    if (toolIds.length === 0) return;
    
    // Batch check existing tools
    const existingMeta = await this.withPerformanceLogging(
      'Batch checking existing tools',
      async () => {
        const multi = this.redisClient.multi();
        for (const toolId of toolIds) {
          multi.get(`bigtool:tools:meta:${toolId}`);
        }
        const results = await multi.exec();
        return results ? results.map((r: any) => r as string | null) : [];
      },
      (_results) => `checked ${toolIds.length} tools`
    );
    
    const newTools: Document[] = [];
    const updateMetaMulti = this.redisClient.multi();
    let unchangedCount = 0;
    
    // Process results and identify new/changed tools
    for (let i = 0; i < toolIds.length; i++) {
      const toolId = toolIds[i];
      const tool = toolRegistry[toolId];
      const existingMetaStr = existingMeta[i];
      const contentHash = this.generateHash(`${tool.name} ${tool.description}`);
      
      let shouldIndex = true;
      if (existingMetaStr) {
        try {
          const meta = JSON.parse(existingMetaStr) as ToolData;
          if (meta.content_hash === contentHash) {
            shouldIndex = false;
            unchangedCount++;
          }
        } catch (e) {
          // Invalid JSON, re-index
        }
      }
      
      if (shouldIndex) {
        const doc = createToolDocument(tool);
        // Add content_hash to metadata
        doc.metadata.content_hash = contentHash;
        newTools.push(doc);
        
        // Add to update pipeline with TTL
        const metaKey = `bigtool:tools:meta:${toolId}`;
        const metaData: ToolData = {
          tool_id: toolId,
          name: tool.name,
          description: tool.description || "",
          content_hash: contentHash,
          indexed_at: Date.now()
        };
        
        updateMetaMulti.set(metaKey, JSON.stringify(metaData));
        updateMetaMulti.expire(metaKey, this.ttlSeconds);
      }
    }
    
    // Bulk update vectors and metadata
    if (newTools.length > 0) {
      await this.withPerformanceLogging(
        'Adding documents to vector store',
        async () => {
          await this.vectorStore.addDocuments(newTools);
          await updateMetaMulti.exec();
        },
        () => `indexed ${newTools.length} documents`
      );
    }
    
    console.log(`Indexed ${newTools.length} new/updated tools, ${unchangedCount} unchanged`);
  }
  
  /**
   * Search for tools using vector similarity
   */
  async search(
    namespacePrefix: string[],
    options?: {
      filter?: Record<string, any>;
      limit?: number;
      offset?: number;
      query?: string;
    }
  ): Promise<SearchItem[]> {
    await this.ensureConnected();
    
    const limit = options?.limit || 10;
    const query = options?.query || "";
    
    if (!query) {
      // No query, return empty results
      return [];
    }
    
    // Perform vector similarity search
    const searchResults = await this.withPerformanceLogging(
      'Vector similarity search',
      () => this.vectorStore.similaritySearchWithScore(query, limit),
      (results) => `found ${results.length} results for query: "${query}"`
    );
    
    return searchResults.map(([doc, score]: [Document, number]) => ({
      value: {
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
   * Clear all indexed tools
   */
  async clearTools(): Promise<void> {
    await this.ensureConnected();
    
    const keys = await this.redisClient.keys('bigtool:tools:*');
    if (keys.length > 0) {
      await this.redisClient.del(keys);
    }
    await this.vectorStore.delete({ deleteAll: true });
    
    if (this.verbose) {
      console.info(`[RedisVectorBaseStore] Cleared ${keys.length} tool metadata keys`);
    }
  }
  
  /**
   * Get indexing statistics
   */
  async getIndexStats(): Promise<{ total: number; oldestIndexed?: Date; newestIndexed?: Date }> {
    await this.ensureConnected();
    
    const keys = await this.redisClient.keys('bigtool:tools:meta:*');
    if (keys.length === 0) {
      return { total: 0 };
    }
    
    // Get all metadata to find oldest and newest
    const multi = this.redisClient.multi();
    for (const key of keys) {
      multi.get(key);
    }
    const results = await multi.exec();
    
    let oldest = Infinity;
    let newest = 0;
    
    results?.forEach((r: any) => {
      if (r) {
        try {
          const meta = JSON.parse(r as string) as ToolData;
          if (meta.indexed_at < oldest) oldest = meta.indexed_at;
          if (meta.indexed_at > newest) newest = meta.indexed_at;
        } catch (e) {
          // Skip invalid entries
        }
      }
    });
    
    return {
      total: keys.length,
      oldestIndexed: oldest !== Infinity ? new Date(oldest) : undefined,
      newestIndexed: newest !== 0 ? new Date(newest) : undefined
    };
  }
  
  /**
   * Ensure Redis connection is established
   */
  private async ensureConnected(): Promise<void> {
    if (!this.isConnected) {
      await this.connect();
    }
  }
  
  /**
   * Required BaseStore methods
   */
  
  async get(_namespace: string[], key: string): Promise<Item | null> {
    await this.ensureConnected();
    
    const metaKey = `bigtool:tools:meta:${key}`;
    const metaStr = await this.redisClient.get(metaKey);
    
    if (!metaStr) return null;
    
    try {
      const meta = JSON.parse(metaStr) as ToolData;
      return {
        value: {
          tool_id: meta.tool_id,
          name: meta.name,
          description: meta.description
        },
        key,
        namespace: _namespace,
        createdAt: new Date(meta.indexed_at),
        updatedAt: new Date(meta.indexed_at)
      };
    } catch (e) {
      return null;
    }
  }
  
  async put(_namespace: string[], key: string, value: Record<string, any>): Promise<void> {
    await this.ensureConnected();
    
    const metaKey = `bigtool:tools:meta:${key}`;
    const contentHash = this.generateHash(`${value.name} ${value.description}`);
    
    const metaData: ToolData = {
      tool_id: key,
      name: value.name,
      description: value.description || "",
      content_hash: contentHash,
      indexed_at: Date.now()
    };
    
    await this.redisClient.set(metaKey, JSON.stringify(metaData));
    await this.redisClient.expire(metaKey, this.ttlSeconds);
    
    // Also add to vector store
    if (value.description) {
      const doc = new Document({
        pageContent: `${value.name} ${value.description}`,
        metadata: {
          tool_id: key,
          name: value.name,
          description: value.description,
          content_hash: contentHash
        }
      });
      await this.vectorStore.addDocuments([doc]);
    }
  }
  
  async delete(_namespace: string[], key: string): Promise<void> {
    await this.ensureConnected();
    
    const metaKey = `bigtool:tools:meta:${key}`;
    await this.redisClient.del(metaKey);
    // Note: RedisVectorStore doesn't support individual document deletion
  }
  
  async list(_namespace: string[]): Promise<Item[]> {
    await this.ensureConnected();
    
    const keys = await this.redisClient.keys('bigtool:tools:meta:*');
    const results: Item[] = [];
    
    if (keys.length === 0) return results;
    
    const multi = this.redisClient.multi();
    for (const key of keys) {
      multi.get(key);
    }
    const values = await multi.exec();
    
    values?.forEach((v: any) => {
      if (v) {
        try {
          const meta = JSON.parse(v as string) as ToolData;
          results.push({
            value: {
              tool_id: meta.tool_id,
              name: meta.name,
              description: meta.description
            },
            key: meta.tool_id,
            namespace: _namespace,
            createdAt: new Date(meta.indexed_at),
            updatedAt: new Date(meta.indexed_at)
          });
        } catch (e) {
          // Skip invalid entries
        }
      }
    });
    
    return results;
  }
  
  async batch<Op extends Operation[]>(
    operations: Op
  ): Promise<OperationResults<Op>> {
    await this.ensureConnected();
    
    const results: any[] = [];
    
    for (const op of operations) {
      if ('value' in op && 'namespace' in op && 'key' in op) {
        // Put operation
        await this.put(op.namespace, op.key, op.value as Record<string, any>);
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
    await this.ensureConnected();
    yield;
  }
}