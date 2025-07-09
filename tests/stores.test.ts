import { describe, it, expect, beforeEach, jest } from "@jest/globals";
import { HNSWLibStore } from "../src/stores/HNSWLibStore.js";
import { Embeddings } from "@langchain/core/embeddings";
import { HNSWLib } from "@langchain/community/vectorstores/hnswlib";
import { Document } from "@langchain/core/documents";
import type { ToolRegistry } from "../src/types.js";

// Mock HNSWLib
jest.mock("@langchain/community/vectorstores/hnswlib");

// Mock embeddings
class MockEmbeddings extends Embeddings {
  async embedDocuments(documents: string[]): Promise<number[][]> {
    return documents.map(() => [0.1, 0.2, 0.3]);
  }
  
  async embedQuery(document: string): Promise<number[]> {
    return [0.1, 0.2, 0.3];
  }
}

describe("HNSWLibStore", () => {
  let store: HNSWLibStore;
  let mockEmbeddings: MockEmbeddings;
  let mockVectorStore: jest.Mocked<HNSWLib>;
  
  beforeEach(() => {
    jest.clearAllMocks();
    mockEmbeddings = new MockEmbeddings();
    
    // Mock HNSWLib constructor and methods
    mockVectorStore = {
      addDocuments: jest.fn().mockResolvedValue(undefined),
      similaritySearchWithScore: jest.fn().mockResolvedValue([])
    } as any;
    
    (HNSWLib as jest.MockedClass<typeof HNSWLib>).mockImplementation(() => mockVectorStore);
  });

  describe("Constructor", () => {
    it("should create store with embeddings", () => {
      store = new HNSWLibStore(mockEmbeddings);
      
      expect(HNSWLib).toHaveBeenCalledWith(mockEmbeddings, {
        space: 'cosine'
      });
    });

    it("should create store with custom config", () => {
      store = new HNSWLibStore(mockEmbeddings, {
        space: 'l2',
        numDimensions: 512
      });
      
      expect(HNSWLib).toHaveBeenCalledWith(mockEmbeddings, {
        space: 'l2',
        numDimensions: 512
      });
    });
  });

  describe("indexTools", () => {
    beforeEach(() => {
      store = new HNSWLibStore(mockEmbeddings);
    });

    it("should index tools from registry", async () => {
      const toolRegistry: ToolRegistry = {
        tool1: {
          name: "Tool One",
          description: "First tool description",
          invoke: jest.fn()
        } as any,
        tool2: {
          name: "Tool Two",
          description: "Second tool description",
          invoke: jest.fn()
        } as any
      };

      await store.indexTools(toolRegistry);

      expect(mockVectorStore.addDocuments).toHaveBeenCalledTimes(1);
      const documents = mockVectorStore.addDocuments.mock.calls[0][0];
      
      expect(documents).toHaveLength(2);
      expect(documents[0].pageContent).toBe("Tool One First tool description");
      expect(documents[0].metadata).toEqual({
        tool_id: "tool1",
        name: "Tool One",
        description: "First tool description"
      });
    });

    it("should handle tools without descriptions", async () => {
      const toolRegistry: ToolRegistry = {
        tool1: {
          name: "Tool One",
          invoke: jest.fn()
        } as any
      };

      await store.indexTools(toolRegistry);

      const documents = mockVectorStore.addDocuments.mock.calls[0][0];
      expect(documents[0].pageContent).toBe("Tool One ");
      expect(documents[0].metadata.description).toBe("");
    });
  });

  describe("search", () => {
    beforeEach(() => {
      store = new HNSWLibStore(mockEmbeddings);
    });

    it("should search with query", async () => {
      const mockResults: [Document, number][] = [
        [
          new Document({
            pageContent: "Tool One",
            metadata: { tool_id: "tool1", name: "Tool One", description: "Test" }
          }),
          0.9
        ]
      ];
      
      mockVectorStore.similaritySearchWithScore.mockResolvedValueOnce(mockResults);
      
      // First index a tool so metadata is available
      await store.put(["tools"], "tool1", {
        tool_id: "tool1",
        name: "Tool One",
        description: "Test"
      });

      const results = await store.search(["tools"], {
        query: "find tool",
        limit: 5
      });

      expect(mockVectorStore.similaritySearchWithScore).toHaveBeenCalledWith("find tool", 5);
      expect(results).toHaveLength(1);
      expect(results[0].value).toEqual({
        tool_id: "tool1",
        name: "Tool One",
        description: "Test"
      });
      expect(results[0].score).toBe(0.9);
    });

    it("should return all tools when no query provided", async () => {
      // Add some tools to metadata
      await store.put(["tools"], "tool1", {
        tool_id: "tool1",
        name: "Tool One",
        description: "First"
      });
      await store.put(["tools"], "tool2", {
        tool_id: "tool2",
        name: "Tool Two",
        description: "Second"
      });

      const results = await store.search(["tools"], { limit: 10 });

      expect(mockVectorStore.similaritySearchWithScore).not.toHaveBeenCalled();
      expect(results).toHaveLength(2);
      expect(results[0].score).toBe(1);
    });

    it("should respect limit when no query", async () => {
      // Add 3 tools
      for (let i = 1; i <= 3; i++) {
        await store.put(["tools"], `tool${i}`, {
          tool_id: `tool${i}`,
          name: `Tool ${i}`,
          description: `Description ${i}`
        });
      }

      const results = await store.search(["tools"], { limit: 2 });

      expect(results).toHaveLength(2);
    });
  });

  describe("BaseStore methods", () => {
    beforeEach(() => {
      store = new HNSWLibStore(mockEmbeddings);
    });

    describe("get", () => {
      it("should get existing item", async () => {
        await store.put(["tools"], "tool1", {
          tool_id: "tool1",
          name: "Tool One",
          description: "Test"
        });

        const item = await store.get(["tools"], "tool1");
        
        expect(item).not.toBeNull();
        expect(item?.value).toEqual({
          tool_id: "tool1",
          name: "Tool One",
          description: "Test"
        });
      });

      it("should return null for non-existent item", async () => {
        const item = await store.get(["tools"], "nonexistent");
        expect(item).toBeNull();
      });
    });

    describe("put", () => {
      it("should store item and add to vector store", async () => {
        const toolData = {
          tool_id: "tool1",
          name: "Tool One",
          description: "Test description"
        };

        await store.put(["tools"], "tool1", toolData);

        expect(mockVectorStore.addDocuments).toHaveBeenCalledTimes(1);
        const documents = mockVectorStore.addDocuments.mock.calls[0][0];
        expect(documents[0].pageContent).toBe("Tool One Test description");
        
        // Verify it can be retrieved
        const item = await store.get(["tools"], "tool1");
        expect(item?.value).toEqual(toolData);
      });

      it("should store item without description", async () => {
        const toolData = {
          tool_id: "tool1",
          name: "Tool One",
          description: ""
        };

        await store.put(["tools"], "tool1", toolData);

        // Should not add to vector store if no description
        expect(mockVectorStore.addDocuments).not.toHaveBeenCalled();
        
        // But should still be retrievable
        const item = await store.get(["tools"], "tool1");
        expect(item?.value).toEqual(toolData);
      });
    });

    describe("delete", () => {
      it("should delete item from metadata", async () => {
        await store.put(["tools"], "tool1", {
          tool_id: "tool1",
          name: "Tool One",
          description: "Test"
        });

        await store.delete(["tools"], "tool1");

        const item = await store.get(["tools"], "tool1");
        expect(item).toBeNull();
      });
    });

    describe("list", () => {
      it("should list all items", async () => {
        await store.put(["tools"], "tool1", {
          tool_id: "tool1",
          name: "Tool One",
          description: "First"
        });
        await store.put(["tools"], "tool2", {
          tool_id: "tool2",
          name: "Tool Two",
          description: "Second"
        });

        const items = await store.list(["tools"]);

        expect(items).toHaveLength(2);
        expect(items.map(i => i.key).sort()).toEqual(["tool1", "tool2"]);
      });

      it("should return empty list when no items", async () => {
        const items = await store.list(["tools"]);
        expect(items).toHaveLength(0);
      });
    });

    describe("batch", () => {
      it("should handle put operations", async () => {
        const operations = [
          {
            namespace: ["tools"],
            key: "tool1",
            value: {
              tool_id: "tool1",
              name: "Tool One",
              description: "First"
            }
          },
          {
            namespace: ["tools"],
            key: "tool2",
            value: {
              tool_id: "tool2",
              name: "Tool Two",
              description: "Second"
            }
          }
        ];

        const results = await store.batch(operations);

        expect(results).toHaveLength(2);
        expect(results[0]).toBeUndefined();
        expect(results[1]).toBeUndefined();

        // Verify items were stored
        const item1 = await store.get(["tools"], "tool1");
        expect(item1?.value.name).toBe("Tool One");
      });

      it("should handle get operations", async () => {
        await store.put(["tools"], "tool1", {
          tool_id: "tool1",
          name: "Tool One",
          description: "Test"
        });

        const operations = [
          { namespace: ["tools"], key: "tool1" },
          { namespace: ["tools"], key: "nonexistent" }
        ];

        const results = await store.batch(operations);

        expect(results).toHaveLength(2);
        expect(results[0]?.value.name).toBe("Tool One");
        expect(results[1]).toBeNull();
      });
    });

    describe("start", () => {
      it("should yield once", async () => {
        const generator = store.start();
        const result = await generator.next();
        
        expect(result.done).toBe(false);
        expect(result.value).toBeUndefined();
        
        const done = await generator.next();
        expect(done.done).toBe(true);
      });
    });
  });
});