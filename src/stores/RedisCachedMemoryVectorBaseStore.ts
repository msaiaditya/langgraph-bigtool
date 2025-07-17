import { Embeddings } from "@langchain/core/embeddings";
import { Document } from "@langchain/core/documents";
import { StructuredTool, Tool, DynamicStructuredTool } from "@langchain/core/tools";
import { createClient, RedisClientType } from "redis";
import { MemoryVectorBaseStore } from "./MemoryVectorBaseStore.js";
import { getToolId, createToolDocument } from "../utils/registry.js";
import type { ToolRegistry } from "../types.js";

interface CachedEmbedding {
  tool_id: string;
  name: string;
  description: string;
  embedding: number[];
  cached_at: number;
}

interface RedisCachedMemoryVectorBaseStoreConfig {
  redisUrl: string;
  embeddings: Embeddings;
  indexName?: string;
  ttlSeconds?: number;
  verbose?: boolean;
}

/**
 * Redis-cached memory vector store that combines Redis caching with in-memory search.
 * 
 * This store uses Redis to cache computed embeddings and loads them into a
 * MemoryVectorBaseStore for fast semantic search. Embeddings are computed only
 * once and reused across application restarts.
 * 
 * @example
 * ```typescript
 * import { RedisCachedMemoryVectorBaseStore } from "langgraph-bigtool";
 * import { OpenAIEmbeddings } from "@langchain/openai";
 * 
 * const embeddings = new OpenAIEmbeddings({
 *   model: "text-embedding-3-small"
 * });
 * 
 * const store = new RedisCachedMemoryVectorBaseStore({
 *   redisUrl: 'redis://localhost:6379',
 *   embeddings,
 *   ttlSeconds: 7 * 24 * 60 * 60 // 7 days
 * });
 * 
 * await store.connect();
 * const agent = await createAgent({ llm, tools, store });
 * ```
 */
export class RedisCachedMemoryVectorBaseStore extends MemoryVectorBaseStore {
  private redisClient!: RedisClientType;
  private redisUrl: string;
  private indexName: string;
  private ttlSeconds?: number;
  private isConnected = false;

  constructor(config: RedisCachedMemoryVectorBaseStoreConfig) {
    super(config.embeddings, { verbose: config.verbose });
    this.redisUrl = config.redisUrl;
    this.indexName = config.indexName || "embeddings";
    this.ttlSeconds = config.ttlSeconds;
  }

  /**
   * Connect to Redis
   */
  async connect(): Promise<void> {
    if (this.isConnected) return;

    this.redisClient = createClient({ url: this.redisUrl });
    await this.redisClient.connect();
    this.isConnected = true;

    if (this.verbose) {
      console.info(`[RedisCachedMemoryVectorBaseStore] Connected to Redis at ${this.redisUrl}`);
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
      console.info(`[RedisCachedMemoryVectorBaseStore] Disconnected from Redis`);
    }
  }

  /**
   * Generate cache key for a tool
   * @param tool - The tool object
   */
  private getCacheKey(tool: StructuredTool | Tool | DynamicStructuredTool): string {
    return `${this.indexName}:${getToolId(tool)}`;
  }

  /**
   * Batch retrieve cached embeddings from Redis
   */
  private async getCachedEmbeddingsBatch(
    tools: Array<[string, StructuredTool | Tool | DynamicStructuredTool]>
  ): Promise<Map<string, CachedEmbedding>> {
    if (tools.length === 0) return new Map();

    // Build cache keys using getCacheKey
    const cacheKeys = tools.map(([_, tool]) => this.getCacheKey(tool));
    const toolIds = tools.map(([id]) => id);
    
    // Batch retrieve using MGET
    const values = await this.redisClient.mGet(cacheKeys);
    
    const results = new Map<string, CachedEmbedding>();
    toolIds.forEach((toolId, index) => {
      const value = values[index];
      if (value) {
        try {
          const parsed = JSON.parse(value) as CachedEmbedding;
          results.set(toolId, parsed);
        } catch (e) {
          // Invalid cache entry, skip
          if (this.verbose) {
            console.warn(`[RedisCachedMemoryVectorBaseStore] Invalid cache entry for ${toolId}`);
          }
        }
      }
    });

    return results;
  }

  /**
   * Batch store embeddings in Redis
   */
  private async cacheEmbeddingsBatch(
    entries: Array<{
      tool: StructuredTool | Tool | DynamicStructuredTool;
      data: CachedEmbedding;
    }>
  ): Promise<void> {
    if (entries.length === 0) return;

    // Use Redis pipeline for batch operations
    const pipeline = this.redisClient.multi();
    
    for (const { tool, data } of entries) {
      const cacheKey = this.getCacheKey(tool);
      const value = JSON.stringify(data);
      
      if (this.ttlSeconds) {
        pipeline.setEx(cacheKey, this.ttlSeconds, value);
      } else {
        pipeline.set(cacheKey, value);
      }
    }

    await pipeline.exec();
  }

  /**
   * Index tools with Redis caching support
   */
  async indexTools(toolRegistry: ToolRegistry): Promise<void> {
    if (!this.isConnected) {
      throw new Error("Redis client not connected. Call connect() first.");
    }

    const startTime = performance.now();
    const tools = Object.entries(toolRegistry);

    // Step 1: Batch retrieve cached embeddings
    const cachedResults = await this.getCachedEmbeddingsBatch(tools);

    // Step 2: Separate cache hits and misses
    const cachedEmbeddings: Array<{
      vector: number[];
      document: Document;
    }> = [];
    const toolsNeedingEmbeddings: Array<{
      toolId: string;
      tool: any;
    }> = [];

    for (const [toolId, tool] of tools) {
      const cached = cachedResults.get(toolId);
      
      if (cached && cached.embedding) {
        // Cache hit - use cached embedding
        cachedEmbeddings.push({
          vector: cached.embedding,
          document: createToolDocument(tool)
        });
      } else {
        // Cache miss - need to compute embedding
        toolsNeedingEmbeddings.push({ toolId: getToolId(tool), tool });
      }
    }

    // Step 3: Compute embeddings for cache misses
    if (toolsNeedingEmbeddings.length > 0) {
      const texts = toolsNeedingEmbeddings.map(({ tool }) =>
        `${tool.name} ${tool.description || ""}`
      );
      
      // Batch compute embeddings
      const embeddings = await this.embeddings.embedDocuments(texts);

      // Prepare entries for batch caching
      const cacheEntries: Array<{
        tool: StructuredTool | Tool | DynamicStructuredTool;
        data: CachedEmbedding;
      }> = [];

      toolsNeedingEmbeddings.forEach(({ tool }, index) => {
        const embedding = embeddings[index];

        // Prepare cache entry
        cacheEntries.push({
          tool,
          data: {
            tool_id: getToolId(tool),
            name: tool.name,
            description: tool.description || "",
            embedding,
            cached_at: Date.now()
          }
        });

        // Add to embeddings array
        cachedEmbeddings.push({
          vector: embedding,
          document: createToolDocument(tool)
        });
      });

      // Step 4: Batch store new embeddings in Redis
      await this.cacheEmbeddingsBatch(cacheEntries);
    }

    // Step 5: Add all vectors to memory store
    if (cachedEmbeddings.length > 0) {
      const vectors = cachedEmbeddings.map(e => e.vector);
      const documents = cachedEmbeddings.map(e => e.document);
      await this.addVectors(vectors, documents);
    }

    if (this.verbose) {
      const elapsed = performance.now() - startTime;
      const hitRate = cachedResults.size / tools.length * 100;
      console.info(
        `[RedisCachedMemoryVectorBaseStore] Indexed ${tools.length} tools in ${elapsed.toFixed(2)}ms ` +
        `(${cachedResults.size} cache hits [${hitRate.toFixed(1)}%], ${toolsNeedingEmbeddings.length} computed)`
      );
    }
  }

  /**
   * Override start to ensure Redis connection
   */
  async *start(): AsyncGenerator<void> {
    if (!this.isConnected) {
      await this.connect();
    }
    yield;
  }
}