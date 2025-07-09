import { describe, it, expect, beforeEach, jest } from "@jest/globals";
import { HTTPEmbeddings } from "../src/embeddings/http.js";

// Mock fetch globally
global.fetch = jest.fn() as jest.MockedFunction<typeof fetch>;

describe("HTTPEmbeddings", () => {
  let embeddings: HTTPEmbeddings;
  const mockFetch = global.fetch as jest.MockedFunction<typeof fetch>;
  const originalWarn = console.warn;
  
  beforeEach(() => {
    jest.clearAllMocks();
    // Reset environment variables
    delete process.env.EMBEDDINGS_SERVICE_URL;
    // Suppress console.warn during tests
    console.warn = jest.fn();
  });

  afterEach(() => {
    // Restore console.warn
    console.warn = originalWarn;
  });

  describe("Constructor", () => {
    it("should use default service URL", () => {
      embeddings = new HTTPEmbeddings();
      expect(embeddings.getServiceUrl()).toBe("http://localhost:8001");
    });

    it("should use provided service URL", () => {
      embeddings = new HTTPEmbeddings({ serviceUrl: "http://custom:3000" });
      expect(embeddings.getServiceUrl()).toBe("http://custom:3000");
    });

    it("should use environment variable if set", () => {
      process.env.EMBEDDINGS_SERVICE_URL = "http://env:4000";
      embeddings = new HTTPEmbeddings();
      expect(embeddings.getServiceUrl()).toBe("http://env:4000");
    });

    it("should prefer config over environment variable", () => {
      process.env.EMBEDDINGS_SERVICE_URL = "http://env:4000";
      embeddings = new HTTPEmbeddings({ serviceUrl: "http://custom:3000" });
      expect(embeddings.getServiceUrl()).toBe("http://custom:3000");
    });
  });

  describe("embedDocuments", () => {
    beforeEach(() => {
      embeddings = new HTTPEmbeddings({ serviceUrl: "http://test:8001" });
    });

    it("should successfully embed multiple documents", async () => {
      const mockResponse = {
        embeddings: [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
        model: "test-model",
        dimensions: 3
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => mockResponse
      } as Response);

      const documents = ["Hello world", "Test document"];
      const result = await embeddings.embedDocuments(documents);

      expect(mockFetch).toHaveBeenCalledWith("http://test:8001/embed", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({ texts: documents })
      });

      expect(result).toEqual(mockResponse.embeddings);
    });

    it("should handle empty documents array", async () => {
      const mockResponse = {
        embeddings: []
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => mockResponse
      } as Response);

      const result = await embeddings.embedDocuments([]);
      expect(result).toEqual([]);
    });

    it("should throw error on non-ok response", async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 500,
        statusText: "Internal Server Error"
      } as Response);

      await expect(embeddings.embedDocuments(["test"]))
        .rejects
        .toThrow("Embeddings service error: 500 Internal Server Error");
    });

    it("should throw error on network failure", async () => {
      mockFetch.mockRejectedValueOnce(new Error("Network error"));

      await expect(embeddings.embedDocuments(["test"]))
        .rejects
        .toThrow("Network error");
    });
  });

  describe("embedQuery", () => {
    beforeEach(() => {
      embeddings = new HTTPEmbeddings({ serviceUrl: "http://test:8001" });
    });

    it("should embed a single query", async () => {
      const mockResponse = {
        embeddings: [[0.1, 0.2, 0.3]]
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => mockResponse
      } as Response);

      const result = await embeddings.embedQuery("Hello world");

      expect(mockFetch).toHaveBeenCalledWith("http://test:8001/embed", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({ texts: ["Hello world"] })
      });

      expect(result).toEqual([0.1, 0.2, 0.3]);
    });

    it("should handle empty query", async () => {
      const mockResponse = {
        embeddings: [[]]
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => mockResponse
      } as Response);

      const result = await embeddings.embedQuery("");
      expect(result).toEqual([]);
    });
  });

  describe("checkHealth", () => {
    beforeEach(() => {
      embeddings = new HTTPEmbeddings({ serviceUrl: "http://test:8001" });
    });

    it("should return true when service is healthy", async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true
      } as Response);

      const result = await embeddings.checkHealth();
      
      expect(mockFetch).toHaveBeenCalledWith("http://test:8001/health");
      expect(result).toBe(true);
    });

    it("should return false when service returns non-ok status", async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false
      } as Response);

      const result = await embeddings.checkHealth();
      expect(result).toBe(false);
    });

    it("should return false on network error", async () => {
      mockFetch.mockRejectedValueOnce(new Error("Network error"));

      const result = await embeddings.checkHealth();
      expect(result).toBe(false);
    });
  });
});