/**
 * FaceForge AI - Dashboard View Controller
 * Loads real application stats and recent generation history from backend.
 */

document.addEventListener("DOMContentLoaded", async () => {
  const statTotalGen = document.getElementById("statTotalGen");
  const statAvgDuration = document.getElementById("statAvgDuration");
  const statSavedCount = document.getElementById("statSavedCount");
  const recentGrid = document.getElementById("recentGenerationsGrid");

  try {
    const res = await window.API.get("/api/dashboard/stats");
    const stats = res.stats;

    if (statTotalGen) statTotalGen.textContent = stats.total_generations.toLocaleString();
    if (statAvgDuration) statAvgDuration.textContent = `${stats.avg_duration_ms} ms`;
    if (statSavedCount) statSavedCount.textContent = stats.saved_count.toLocaleString();

    if (recentGrid) {
      if (!stats.recent_generations || stats.recent_generations.length === 0) {
        recentGrid.innerHTML = `
          <div class="empty-state" style="grid-column: 1 / -1;">
            <div class="empty-state-icon">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <rect x="3" y="3" width="18" height="18" rx="2" ry="2"/>
                <circle cx="8.5" cy="8.5" r="1.5"/>
                <polyline points="21 15 16 10 5 21"/>
              </svg>
            </div>
            <h3>No Generations Yet</h3>
            <p>Generate your first synthetic human face using the DCGAN neural network.</p>
            <a href="/studio" class="btn btn-primary" style="margin-top: 1rem;">Launch Face Studio</a>
          </div>
        `;
      } else {
        recentGrid.innerHTML = stats.recent_generations.map(item => `
          <div class="recent-thumb-card" onclick="window.location.href='/gallery'">
            <img class="recent-thumb-img" src="${item.image_url}" alt="Seed ${item.seed}" loading="lazy"/>
            <div class="recent-thumb-info">
              <span>Seed: <strong>${item.seed}</strong></span>
              <span>${item.duration_ms}ms</span>
            </div>
          </div>
        `).join("");
      }
    }
  } catch (err) {
    console.error("Failed to load dashboard metrics:", err);
  }
});
