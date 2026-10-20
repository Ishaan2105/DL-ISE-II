/**
 * FaceForge AI - Model Explorer View Controller
 * Interactively displays Generator and Discriminator neural network structures,
 * tensor dimensions, layer specifications, and loss function mathematics.
 */

document.addEventListener("DOMContentLoaded", async () => {
  const tabGenerator = document.getElementById("tabGenerator");
  const tabDiscriminator = document.getElementById("tabDiscriminator");
  const layerTableBody = document.getElementById("layerTableBody");
  const archTotalParams = document.getElementById("archTotalParams");
  const archTrainableParams = document.getElementById("archTrainableParams");
  const archResolution = document.getElementById("archResolution");
  const archLatentDim = document.getElementById("archLatentDim");
  const btnDownloadCkpt = document.getElementById("btnDownloadCkpt");
  const ckptStatusText = document.getElementById("ckptStatusText");

  let modelData = null;
  let currentTab = "generator";

  try {
    const res = await window.API.get("/api/model/info");
    modelData = res.model;
    renderModelInfo();
  } catch (err) {
    console.error("Failed to load model info:", err);
    window.UI.showToast("Could not retrieve model specifications.", "error");
  }

  function renderModelInfo() {
    if (!modelData) return;

    if (archResolution) archResolution.textContent = `${modelData.native_resolution}x${modelData.native_resolution}`;
    if (archLatentDim) archLatentDim.textContent = `${modelData.latent_dim} dims`;

    if (ckptStatusText) {
      if (modelData.is_checkpoint_loaded) {
        ckptStatusText.innerHTML = `<span class="badge badge-success">Pretrained CelebA Loaded</span>`;
        if (btnDownloadCkpt) btnDownloadCkpt.style.display = "none";
      } else {
        ckptStatusText.innerHTML = `<span class="badge badge-warning">Random Initial Weights</span>`;
        if (btnDownloadCkpt) btnDownloadCkpt.style.display = "inline-flex";
      }
    }

    renderLayers();
  }

  function renderLayers() {
    if (!modelData) return;
    const summary = currentTab === "generator" ? modelData.generator_summary : modelData.discriminator_summary;
    if (!summary) return;

    if (archTotalParams) archTotalParams.textContent = summary.total_parameters.toLocaleString();
    if (archTrainableParams) archTrainableParams.textContent = summary.trainable_parameters.toLocaleString();

    if (layerTableBody) {
      layerTableBody.innerHTML = summary.layers.map(layer => `
        <tr>
          <td><strong style="color: var(--accent-highlight);">${layer.index}</strong></td>
          <td><strong>${layer.name}</strong></td>
          <td><span class="badge badge-cyan">${layer.type}</span></td>
          <td><code class="font-mono text-cyan">${layer.output_shape}</code></td>
          <td class="font-mono">${layer.params > 0 ? layer.params.toLocaleString() : "—"}</td>
          <td><span class="badge badge-violet">${layer.activation}</span></td>
          <td style="color: var(--text-muted); font-size: 0.8rem;">${layer.notes}</td>
        </tr>
      `).join("");
    }
  }

  if (tabGenerator) {
    tabGenerator.addEventListener("click", () => {
      tabGenerator.classList.add("active");
      tabDiscriminator.classList.remove("active");
      currentTab = "generator";
      renderLayers();
    });
  }

  if (tabDiscriminator) {
    tabDiscriminator.addEventListener("click", () => {
      tabDiscriminator.classList.add("active");
      tabGenerator.classList.remove("active");
      currentTab = "discriminator";
      renderLayers();
    });
  }

  if (btnDownloadCkpt) {
    btnDownloadCkpt.addEventListener("click", async () => {
      btnDownloadCkpt.disabled = true;
      btnDownloadCkpt.innerHTML = `<span class="spinner"></span> Downloading Pretrained Weights...`;
      try {
        const res = await window.API.post("/api/model/download-checkpoint");
        window.UI.showToast(res.message, "success");
        setTimeout(() => window.location.reload(), 1500);
      } catch (err) {
        window.UI.showToast(`Download failed: ${err.message}`, "error");
        btnDownloadCkpt.disabled = false;
        btnDownloadCkpt.innerHTML = `Download CelebA Checkpoint`;
      }
    });
  }
});
