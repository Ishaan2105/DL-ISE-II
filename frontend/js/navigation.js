/**
 * FaceForge AI - Global Navigation & System Status Poller
 */

document.addEventListener("DOMContentLoaded", () => {
  // Highlight active navigation link
  const currentPath = window.location.pathname;
  const navItems = document.querySelectorAll(".nav-item");

  navItems.forEach(item => {
    const href = item.getAttribute("href");
    if (href === currentPath || (currentPath === "/" && (href === "/dashboard" || href === "/index.html" || href === "/"))) {
      item.classList.add("active");
    } else if (href && currentPath.includes(href) && href !== "/" && href !== "/dashboard") {
      item.classList.add("active");
    } else {
      item.classList.remove("active");
    }
  });

  // Mobile menu drawer toggle
  const mobileMenuBtn = document.getElementById("mobileMenuBtn");
  const sidebar = document.querySelector(".app-sidebar");

  if (mobileMenuBtn && sidebar) {
    mobileMenuBtn.addEventListener("click", () => {
      sidebar.classList.toggle("open");
    });

    // Close when clicking outside
    document.addEventListener("click", (e) => {
      if (!sidebar.contains(e.target) && !mobileMenuBtn.contains(e.target) && sidebar.classList.contains("open")) {
        sidebar.classList.remove("open");
      }
    });
  }

  // Poll system health
  async function updateHealth() {
    try {
      const data = await window.API.get("/api/health");
      const devicePill = document.getElementById("systemDeviceBadge");
      const statusPill = document.getElementById("systemStatusText");

      if (devicePill) {
        devicePill.innerHTML = `<span class="dot"></span> ${data.device || "CPU"}`;
      }

      if (statusPill) {
        statusPill.textContent = data.checkpoint_loaded ? "Model Ready" : "Initial Weights";
      }
    } catch (err) {
      const statusPill = document.getElementById("systemStatusText");
      if (statusPill) {
        statusPill.textContent = "Offline";
        statusPill.style.color = "var(--color-error)";
      }
    }
  }

  updateHealth();
  setInterval(updateHealth, window.CONFIG?.POLL_INTERVAL_HEALTH_MS || 15000);
});
