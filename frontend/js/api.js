/**
 * FaceForge AI - Centralized REST API Utility
 * Provides asynchronous network requests with JSON parsing, error normalization,
 * status validation, and offline recovery handling.
 */

const API = {
  async request(endpoint, options = {}) {
    const url = `${window.CONFIG?.API_BASE_URL || ""}${endpoint}`;
    const defaultHeaders = {
      "Accept": "application/json"
    };

    if (options.body && typeof options.body === "object" && !(options.body instanceof FormData)) {
      options.body = JSON.stringify(options.body);
      defaultHeaders["Content-Type"] = "application/json";
    }

    const config = {
      ...options,
      headers: {
        ...defaultHeaders,
        ...options.headers
      }
    };

    try {
      const response = await fetch(url, config);
      const isJson = response.headers.get("content-type")?.includes("application/json");
      const data = isJson ? await response.json() : await response.text();

      if (!response.ok) {
        const errorMsg = (data && data.error) || (data && data.message) || `HTTP error ${response.status}`;
        throw new Error(errorMsg);
      }

      return data;
    } catch (err) {
      console.error(`[API Error] ${options.method || "GET"} ${url}:`, err);
      throw err;
    }
  },

  get(endpoint, params = null) {
    let url = endpoint;
    if (params) {
      const query = new URLSearchParams(
        Object.entries(params).filter(([_, v]) => v !== null && v !== undefined && v !== "")
      ).toString();
      if (query) url += `?${query}`;
    }
    return this.request(url, { method: "GET" });
  },

  post(endpoint, body = {}) {
    return this.request(endpoint, { method: "POST", body });
  },

  delete(endpoint) {
    return this.request(endpoint, { method: "DELETE" });
  }
};

window.API = API;
