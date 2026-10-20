/**
 * FaceForge AI - Gallery View Controller
 * Manages search, model filters, sort orders, detail inspection modal,
 * and safe deletion procedures.
 */

document.addEventListener("DOMContentLoaded", () => {
  const galleryGrid = document.getElementById("galleryGrid");
  const gallerySearchInput = document.getElementById("gallerySearchInput");
  const gallerySortSelect = document.getElementById("gallerySortSelect");
  const galleryModelFilter = document.getElementById("galleryModelFilter");
  const toggleSavedOnly = document.getElementById("toggleSavedOnly");
  const galleryCountText = document.getElementById("galleryCountText");

  // Detail Modal Elements
  const detailModal = document.getElementById("detailModal");
  const modalImg = document.getElementById("modalImg");
  const modalSeed = document.getElementById("modalSeed");
  const modalRes = document.getElementById("modalRes");
  const modalDuration = document.getElementById("modalDuration");
  const modalDevice = document.getElementById("modalDevice");
  const modalDate = document.getElementById("modalDate");
  const modalId = document.getElementById("modalId");
  const btnModalDownload = document.getElementById("btnModalDownload");
  const btnModalDelete = document.getElementById("btnModalDelete");

  // Delete Confirmation Modal
  const confirmDeleteModal = document.getElementById("confirmDeleteModal");
  const btnConfirmDelete = document.getElementById("btnConfirmDelete");
  const btnCancelDelete = document.getElementById("btnCancelDelete");

  let currentItems = [];
  let pendingDeleteId = null;

  async function loadGallery() {
    const search = gallerySearchInput?.value || "";
    const sort = gallerySortSelect?.value || "newest";
    const model = galleryModelFilter?.value || "all";
    const savedOnly = toggleSavedOnly?.checked || false;

    try {
      const res = await window.API.get("/api/gallery", {
        search,
        sort,
        model,
        saved_only: savedOnly,
        limit: 50
      });

      const data = res.data;
      currentItems = data.items;

      if (galleryCountText) {
        galleryCountText.textContent = `${data.total} item(s) found`;
      }

      renderGallery(currentItems);
    } catch (err) {
      console.error("Failed to load gallery:", err);
      window.UI.showToast("Failed to load gallery records.", "error");
    }
  }

  function renderGallery(items) {
    if (!galleryGrid) return;

    if (!items || items.length === 0) {
      galleryGrid.innerHTML = `
        <div class="empty-state" style="grid-column: 1 / -1;">
          <div class="empty-state-icon">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <rect x="3" y="3" width="18" height="18" rx="2" ry="2"/>
              <circle cx="8.5" cy="8.5" r="1.5"/>
              <polyline points="21 15 16 10 5 21"/>
            </svg>
          </div>
          <h3>No Images Found</h3>
          <p>No synthetic faces match your current filter criteria.</p>
        </div>
      `;
      return;
    }

    galleryGrid.innerHTML = items.map(item => `
      <div class="gallery-card" data-id="${item.id}">
        <div class="gallery-img-wrapper">
          <img class="gallery-img" src="${item.image_url}" alt="Seed ${item.seed}" loading="lazy"/>
          <div class="gallery-card-overlay">
            <button class="btn btn-primary btn-sm view-btn" data-id="${item.id}">Inspect</button>
            <div class="overlay-actions">
              <button class="btn btn-icon dl-btn" data-url="${item.image_url}" data-name="${item.filename}" title="Download PNG">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>
              </button>
              <button class="btn btn-icon btn-danger del-btn" data-id="${item.id}" title="Delete">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/></svg>
              </button>
            </div>
          </div>
        </div>
        <div class="gallery-info">
          <div class="gallery-title">
            <span>Seed: <strong>${item.seed}</strong></span>
            ${item.saved_to_gallery ? `<span class="badge badge-cyan">Saved</span>` : ""}
          </div>
          <div class="gallery-meta">
            <span>${item.resolution}x${item.resolution}</span>
            <span>${item.duration_ms}ms</span>
          </div>
        </div>
      </div>
    `).join("");

    // Attach Click Handlers
    galleryGrid.querySelectorAll(".view-btn").forEach(btn => {
      btn.addEventListener("click", () => openDetailModal(btn.dataset.id));
    });

    galleryGrid.querySelectorAll(".dl-btn").forEach(btn => {
      btn.addEventListener("click", (e) => {
        e.stopPropagation();
        window.UI.downloadImage(btn.dataset.url, btn.dataset.name);
      });
    });

    galleryGrid.querySelectorAll(".del-btn").forEach(btn => {
      btn.addEventListener("click", (e) => {
        e.stopPropagation();
        requestDelete(btn.dataset.id);
      });
    });
  }

  function openDetailModal(id) {
    const item = currentItems.find(i => i.id === id);
    if (!item) return;

    modalImg.src = item.image_url;
    modalSeed.textContent = item.seed;
    modalRes.textContent = `${item.resolution}x${item.resolution} px`;
    modalDuration.textContent = `${item.duration_ms} ms`;
    modalDevice.textContent = item.device;
    modalDate.textContent = new Date(item.created_at).toLocaleString();
    modalId.textContent = item.id;

    btnModalDownload.onclick = () => window.UI.downloadImage(item.image_url, item.filename);
    btnModalDelete.onclick = () => {
      window.UI.closeModal("detailModal");
      requestDelete(item.id);
    };

    window.UI.openModal("detailModal");
  }

  function requestDelete(id) {
    pendingDeleteId = id;
    window.UI.openModal("confirmDeleteModal");
  }

  if (btnConfirmDelete) {
    btnConfirmDelete.addEventListener("click", async () => {
      if (!pendingDeleteId) return;
      try {
        await window.API.delete(`/api/gallery/${pendingDeleteId}`);
        window.UI.showToast("Image deleted successfully.", "success");
        window.UI.closeModal("confirmDeleteModal");
        pendingDeleteId = null;
        loadGallery();
      } catch (err) {
        window.UI.showToast("Failed to delete image.", "error");
      }
    });
  }

  if (btnCancelDelete) {
    btnCancelDelete.addEventListener("click", () => {
      window.UI.closeModal("confirmDeleteModal");
      pendingDeleteId = null;
    });
  }

  // Filter & Search events with debounce
  let debounceTimer;
  gallerySearchInput?.addEventListener("input", () => {
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(loadGallery, 300);
  });

  gallerySortSelect?.addEventListener("change", loadGallery);
  galleryModelFilter?.addEventListener("change", loadGallery);
  toggleSavedOnly?.addEventListener("change", loadGallery);

  // Initial load
  loadGallery();
});
