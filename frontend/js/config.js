/**
 * FaceForge AI - Frontend Configuration
 */

const CONFIG = {
  API_BASE_URL: "",  // Relative origin when served from Flask
  POLL_INTERVAL_HEALTH_MS: 15000,
  DEFAULT_RESOLUTION: 64,
  DEFAULT_BATCH_SIZE: 1,
  MAX_BATCH_SIZE: 8
};

window.CONFIG = CONFIG;
