import { Embeddings, EmbeddingsParams } from "@langchain/core/embeddings";

export interface HTTPEmbeddingsConfig extends EmbeddingsParams {
  serviceUrl?: string;
  verbose?: boolean;
}

export interface EmbeddingsResponse {
  embeddings: number[][];
  model?: string;
  dimensions?: number;
}

/**
 * HTTP-based embeddings implementation that connects to an external service.
 * Extends LangChain's Embeddings class for compatibility with vector stores.
 */
export class HTTPEmbeddings extends Embeddings {
  private serviceUrl: string;
  private verbose: boolean;
  
  constructor(config: HTTPEmbeddingsConfig = {}) {
    super(config);
    this.serviceUrl = config.serviceUrl || process.env.EMBEDDINGS_SERVICE_URL || 'http://localhost:8001';
    this.verbose = config.verbose || false;
  }
  
  /**
   * Generate embeddings for multiple documents
   */
  async embedDocuments(documents: string[]): Promise<number[][]> {
    const startTime = performance.now();
    
    try {
      const response = await fetch(`${this.serviceUrl}/embed`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ texts: documents }),
      });
      
      if (!response.ok) {
        throw new Error(`Embeddings service error: ${response.status} ${response.statusText}`);
      }
      
      const data = await response.json() as EmbeddingsResponse;
      
      const endTime = performance.now();
      const latency = endTime - startTime;
      
      if (this.verbose) {
        console.info(
          `[HTTPEmbeddings] Generated ${documents.length} embeddings in ${latency.toFixed(2)}ms ` +
          `(${(latency / documents.length).toFixed(2)}ms per document)`
        );
      }
      
      return data.embeddings;
    } catch (error) {
      const endTime = performance.now();
      const latency = endTime - startTime;
      
      console.warn(
        `[HTTPEmbeddings] Failed after ${latency.toFixed(2)}ms. ` +
        `Failed to connect to embeddings service at ${this.serviceUrl}. ` +
        "Make sure the embeddings service is running."
      );
      throw error;
    }
  }
  
  /**
   * Generate embedding for a single query
   */
  async embedQuery(document: string): Promise<number[]> {
    const startTime = performance.now();
    const embeddings = await this.embedDocuments([document]);
    
    if (this.verbose) {
      const endTime = performance.now();
      const latency = endTime - startTime;
      console.info(
        `[HTTPEmbeddings] Generated single query embedding in ${latency.toFixed(2)}ms`
      );
    }
    
    return embeddings[0];
  }
  
  /**
   * Check if embeddings service is available
   */
  async checkHealth(): Promise<boolean> {
    try {
      const response = await fetch(`${this.serviceUrl}/health`);
      return response.ok;
    } catch {
      return false;
    }
  }
  
  /**
   * Get the service URL
   */
  getServiceUrl(): string {
    return this.serviceUrl;
  }
}