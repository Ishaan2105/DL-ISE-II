/**
 * FaceForge AI - Face Studio Controller
 * Handles model inference, batch generation, seed dice, gallery saving,
 * and continuous latent-space interpolation.
 */

document.addEventListener("DOMContentLoaded", () => {
  // Elements
  const seedInput = document.getElementById("seedInput");
  const btnRandomSeed = document.getElementById("btnRandomSeed");
  const batchButtons = document.querySelectorAll(".batch-opt");
  const resButtons = document.querySelectorAll(".res-opt");
  const btnGenerate = document.getElementById("btnGenerate");
  const btnGenerateAnother = document.getElementById("btnGenerateAnother");
  
  const viewportFrame = document.getElementById("viewportFrame");
  const viewportImg = document.getElementById("viewportImg");
  const viewportPlaceholder = document.getElementById("viewportPlaceholder");
  const generatorLoader = document.getElementById("generatorLoader");
  const batchContainer = document.getElementById("batchContainer");
  const batchGrid = document.getElementById("batchGrid");

  // Metadata Display Elements
  const metaBar = document.getElementById("metadataBar");
  const metaSeed = document.getElementById("metaSeed");
  const metaRes = document.getElementById("metaRes");
  const metaDuration = document.getElementById("metaDuration");
  const metaDevice = document.getElementById("metaDevice");
  const metaModel = document.getElementById("metaModel");

  // Actions
  const btnSaveGallery = document.getElementById("btnSaveGallery");
  const btnDownloadPng = document.getElementById("btnDownloadPng");

  // Latent Morpher Elements
  const morphSeed1 = document.getElementById("morphSeed1");
  const morphSeed2 = document.getElementById("morphSeed2");
  const morphSteps = document.getElementById("morphSteps");
  const btnRunMorph = document.getElementById("btnRunMorph");
  const morphResultsContainer = document.getElementById("morphResultsContainer");
  const morphStripImg = document.getElementById("morphStripImg");
  const btnDownloadStrip = document.getElementById("btnDownloadStrip");

  // State
  let currentBatchSize = 1;
  let currentResolution = 64;
  let currentGeneration = null;
  let currentBatchResults = [];

  // Seed Randomizer
  function getRandomSeed() {
    return Math.floor(Math.random() * 1000000);
  }

  if (btnRandomSeed) {
    btnRandomSeed.addEventListener("click", () => {
      seedInput.value = getRandomSeed();
    });
  }

  // Batch Selector Toggle
  batchButtons.forEach(btn => {
    btn.addEventListener("click", () => {
      batchButtons.forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      currentBatchSize = parseInt(btn.dataset.batch, 10);
    });
  });

  // Resolution Selector Toggle
  resButtons.forEach(btn => {
    btn.addEventListener("click", () => {
      resButtons.forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      currentResolution = parseInt(btn.dataset.res, 10);
    });
  });

  // Generate Execution Function
  async function triggerGeneration(seed) {
    if (seed === undefined || seed === null || isNaN(seed) || seed < 0) {
      window.UI.showToast("Please enter a valid non-negative seed number.", "error");
      return;
    }

    // Set UI to loading state
    btnGenerate.disabled = true;
    if (btnGenerateAnother) btnGenerateAnother.disabled = true;
    generatorLoader.classList.add("active");
    viewportFrame.classList.add("loading");

    try {
      const response = await window.API.post("/api/generate", {
        seed: parseInt(seed, 10),
        batch_size: currentBatchSize,
        resolution: currentResolution
      });

      if (!response.success || !response.generations || response.generations.length === 0) {
        throw new Error(response.error || "Generation produced no images.");
      }

      currentBatchResults = response.generations;
      displaySingleGeneration(currentBatchResults[0]);

      // Render Batch Grid if batch_size > 1
      if (currentBatchSize > 1) {
        batchContainer.style.display = "block";
        renderBatchGrid(currentBatchResults);
      } else {
        batchContainer.style.display = "none";
      }

      window.UI.showToast(`Synthesized ${currentBatchResults.length} face(s) in ${currentBatchResults[0].duration_ms}ms!`, "success");
    } catch (err) {
      window.UI.showToast(`Generation failed: ${err.message}`, "error");
    } finally {
      btnGenerate.disabled = false;
      if (btnGenerateAnother) btnGenerateAnother.disabled = false;
      generatorLoader.classList.remove("active");
      viewportFrame.classList.remove("loading");
    }
  }

  function displaySingleGeneration(gen) {
    currentGeneration = gen;
    viewportPlaceholder.style.display = "none";
    viewportImg.style.display = "block";
    viewportImg.src = gen.image_url;

    // Update metadata pills
    metaBar.style.display = "flex";
    if (metaSeed) metaSeed.textContent = gen.seed;
    if (metaRes) metaRes.textContent = `${gen.resolution}x${gen.resolution} ${gen.is_upscaled ? "(Bicubic)" : "(Native)"}`;
    if (metaDuration) metaDuration.textContent = `${gen.duration_ms} ms`;
    if (metaDevice) metaDevice.textContent = gen.device;
    if (metaModel) metaModel.textContent = gen.model_name;

    // Reset save button state
    if (btnSaveGallery) {
      btnSaveGallery.disabled = false;
      btnSaveGallery.innerHTML = `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M19 21l-7-5-7 5V5a2 2 0 0 1 2-2h10a2 2 0 0 1 2 2z"/></svg> Save to Gallery`;
      btnSaveGallery.classList.remove("btn-cyan");
      btnSaveGallery.classList.add("btn-secondary");
    }

    if (btnDownloadPng) {
      btnDownloadPng.disabled = false;
    }
  }

  function renderBatchGrid(results) {
    batchGrid.innerHTML = results.map((item, idx) => `
      <div class="batch-item ${idx === 0 ? "active" : ""}" data-idx="${idx}">
        <img src="${item.image_url}" alt="Batch Item ${idx + 1}" />
      </div>
    `).join("");

    batchGrid.querySelectorAll(".batch-item").forEach(itemEl => {
      itemEl.addEventListener("click", () => {
        batchGrid.querySelectorAll(".batch-item").forEach(b => b.classList.remove("active"));
        itemEl.classList.add("active");
        const idx = parseInt(itemEl.dataset.idx, 10);
        displaySingleGeneration(currentBatchResults[idx]);
      });
    });
  }

  // Trigger buttons
  btnGenerate.addEventListener("click", () => {
    const seed = parseInt(seedInput.value, 10);
    triggerGeneration(seed);
  });

  if (btnGenerateAnother) {
    btnGenerateAnother.addEventListener("click", () => {
      const newSeed = getRandomSeed();
      seedInput.value = newSeed;
      triggerGeneration(newSeed);
    });
  }

  // Save to Gallery
  if (btnSaveGallery) {
    btnSaveGallery.addEventListener("click", async () => {
      if (!currentGeneration) return;
      try {
        await window.API.post("/api/gallery/save", { id: currentGeneration.id, saved: true });
        window.UI.showToast("Saved to your gallery!", "success");
        btnSaveGallery.innerHTML = `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="var(--color-success)" stroke-width="2"><polyline points="20 6 9 17 4 12"/></svg> Saved`;
        btnSaveGallery.classList.remove("btn-secondary");
        btnSaveGallery.classList.add("btn-cyan");
      } catch (err) {
        window.UI.showToast("Failed to save to gallery.", "error");
      }
    });
  }

  // Download PNG
  if (btnDownloadPng) {
    btnDownloadPng.addEventListener("click", () => {
      if (!currentGeneration) return;
      window.UI.downloadImage(currentGeneration.image_url, currentGeneration.filename);
    });
  }

  // Latent Space Interpolation Controller
  if (btnRunMorph) {
    btnRunMorph.addEventListener("click", async () => {
      const s1 = parseInt(morphSeed1.value, 10);
      const s2 = parseInt(morphSeed2.value, 10);
      const steps = parseInt(morphSteps.value, 10);

      btnRunMorph.disabled = true;
      btnRunMorph.innerHTML = `<span class="spinner"></span> Interpolating...`;

      try {
        const res = await window.API.post("/api/generate/interpolate", {
          seed1: s1,
          seed2: s2,
          steps: steps,
          resolution: currentResolution
        });

        const interp = res.interpolation;
        morphResultsContainer.style.display = "block";
        morphStripImg.src = interp.strip_url;

        if (btnDownloadStrip) {
          btnDownloadStrip.onclick = () => {
            window.UI.downloadImage(interp.strip_url, interp.strip_filename);
          };
        }

        window.UI.showToast(`Synthesized ${steps}-frame continuous latent morph!`, "success");
      } catch (err) {
        window.UI.showToast(`Morph failed: ${err.message}`, "error");
      } finally {
        btnRunMorph.disabled = false;
        btnRunMorph.innerHTML = `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="M8 12h8"/><path d="M12 8v8"/></svg> Generate Morph Sequence`;
      }
    });
  }
});
