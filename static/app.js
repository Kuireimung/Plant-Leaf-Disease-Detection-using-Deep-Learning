const fileInput = document.getElementById("fileInput");
const predictButton = document.getElementById("predictButton");
const dropzone = document.getElementById("dropzone");
const previewImage = document.getElementById("previewImage");
const previewPlaceholder = document.getElementById("previewPlaceholder");
const previewStatus = document.getElementById("previewStatus");
const selectedFileName = document.getElementById("selectedFileName");
const resultState = document.getElementById("resultState");
const resultContent = document.getElementById("resultContent");
const resultPanel = document.getElementById("resultPanel");
const resultBadge = document.getElementById("resultBadge");
const resultHeadline = document.getElementById("resultHeadline");
const resultLabelCaption = document.getElementById("resultLabelCaption");
const resultLabel = document.getElementById("resultLabel");
const resultConfidenceCaption = document.getElementById("resultConfidenceCaption");
const resultConfidence = document.getElementById("resultConfidence");
const resultNotes = document.getElementById("resultNotes");
const predictionBars = document.getElementById("predictionBars");
const modelUsed = document.getElementById("modelUsed");

let currentObjectUrl = null;

function clearObjectUrl() {
  if (currentObjectUrl) {
    URL.revokeObjectURL(currentObjectUrl);
    currentObjectUrl = null;
  }
}

function setPreview(file) {
  clearObjectUrl();

  if (!file) {
    previewImage.hidden = true;
    previewImage.removeAttribute("src");
    previewPlaceholder.hidden = false;
    selectedFileName.textContent = "No file selected yet.";
    previewStatus.textContent = "Waiting for image upload";
    return;
  }

  currentObjectUrl = URL.createObjectURL(file);
  previewImage.src = currentObjectUrl;
  previewImage.hidden = false;
  previewPlaceholder.hidden = true;
  selectedFileName.textContent = `Selected file: ${file.name}`;
  previewStatus.textContent = "Preview ready for analysis";
}

function setNeutralResult(message) {
  resultPanel.classList.remove("healthy", "alert", "unsupported", "is-loading");
  resultPanel.classList.add("neutral");
  resultBadge.className = "result-badge neutral";
  resultBadge.textContent = "Awaiting result";
  resultHeadline.textContent = "The model summary will appear here after analysis.";
  resultLabelCaption.textContent = "Top prediction";
  resultConfidenceCaption.textContent = "Confidence";
  resultState.hidden = false;
  resultState.textContent = message;
  resultContent.hidden = true;
  modelUsed.textContent = "Model: -";
  predictionBars.innerHTML = "";
}

function formatPercent(value) {
  return `${(value * 100).toFixed(2)}%`;
}

function renderPredictions(predictions) {
  predictionBars.innerHTML = "";

  predictions.forEach((entry) => {
    const row = document.createElement("div");
    row.className = "prediction-row";

    const label = document.createElement("span");
    label.className = "prediction-name";
    label.textContent = entry.label;

    const track = document.createElement("div");
    track.className = "prediction-track";

    const fill = document.createElement("div");
    fill.className = "prediction-fill";
    fill.style.width = `${Math.max(entry.confidence * 100, 4)}%`;
    track.appendChild(fill);

    const value = document.createElement("span");
    value.className = "prediction-value";
    value.textContent = formatPercent(entry.confidence);

    row.appendChild(label);
    row.appendChild(track);
    row.appendChild(value);
    predictionBars.appendChild(row);
  });
}

function getSelectedModel() {
  const selectedModel = document.querySelector('input[name="modelChoice"]:checked');
  return selectedModel ? selectedModel.value : "resnet18";
}

function getSelectedModelLabel() {
  const selectedModel = document.querySelector('input[name="modelChoice"]:checked');
  const option = selectedModel ? selectedModel.closest(".model-option") : null;
  const label = option ? option.querySelector("strong") : null;
  return label ? label.textContent.trim() : "Selected model";
}

function setLoadingResult() {
  resultPanel.classList.remove("healthy", "alert", "unsupported");
  resultPanel.classList.add("neutral", "is-loading");
  resultBadge.className = "result-badge neutral";
  resultBadge.textContent = "Analyzing";
  resultHeadline.textContent = "The detector is preparing your image and running inference.";
  resultLabelCaption.textContent = "Top prediction";
  resultConfidenceCaption.textContent = "Confidence";
  resultState.hidden = false;
  resultContent.hidden = true;
  modelUsed.textContent = "Model: -";
  predictionBars.innerHTML = "";

  resultState.innerHTML = `
    <div class="loading-diagnosis" role="status" aria-live="polite">
      <div class="loading-status">
        <span class="loading-dot" aria-hidden="true"></span>
        <span>Inference running</span>
      </div>
      <div class="loading-copy">
        <h4>Checking image</h4>
        <p>${getSelectedModelLabel()} is comparing the upload against PlantVillage classes.</p>
      </div>
      <div class="loading-progress" aria-hidden="true"><span></span></div>
      <div class="loading-steps">
        <span><strong>01</strong> Prepare image</span>
        <span><strong>02</strong> Run classifier</span>
        <span><strong>03</strong> Check confidence</span>
      </div>
    </div>
  `;
}

function applyResultTone(payload) {
  if (payload.is_supported_image === false) {
    resultPanel.classList.remove("neutral", "healthy", "alert", "is-loading");
    resultPanel.classList.add("unsupported");

    resultBadge.className = "result-badge unsupported";
    resultBadge.textContent = "Not a diagnosis";
    resultHeadline.textContent = "The detector rejected this image because it does not look like a supported crop leaf.";
    resultLabelCaption.textContent = "Input check";
    resultConfidenceCaption.textContent = "Closest forced match";
    return;
  }

  const isHealthy = payload.label.toLowerCase().includes("healthy");
  resultLabelCaption.textContent = "Top prediction";
  resultConfidenceCaption.textContent = "Confidence";

  resultPanel.classList.remove("neutral", "healthy", "alert", "unsupported", "is-loading");
  resultPanel.classList.add(isHealthy ? "healthy" : "alert");

  resultBadge.className = `result-badge ${isHealthy ? "healthy" : "alert"}`;
  resultBadge.textContent = isHealthy ? "Healthy signal" : "Disease pattern detected";
  resultHeadline.textContent = isHealthy
    ? "The uploaded leaf looks healthy to the model."
    : "The uploaded leaf shows a disease-like pattern according to the model.";
}

async function analyzeLeaf() {
  const file = fileInput.files[0];
  if (!file) {
    setNeutralResult("Please choose a leaf image first.");
    return;
  }

  const formData = new FormData();
  formData.append("file", file);
  formData.append("model", getSelectedModel());

  predictButton.disabled = true;
  predictButton.textContent = "Analyzing...";
  setLoadingResult();

  try {
    const response = await fetch("/predict", {
      method: "POST",
      body: formData,
    });

    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "Prediction failed.");
    }

    applyResultTone(payload);
    resultLabel.textContent = payload.label;
    resultConfidence.textContent = formatPercent(payload.confidence);
    modelUsed.textContent = `Model: ${payload.model_name || "Selected model"}`;
    resultNotes.textContent = payload.notes;
    renderPredictions(payload.top_predictions);

    resultState.hidden = true;
    resultContent.hidden = false;
  } catch (error) {
    setNeutralResult(error.message || "Something went wrong while predicting.");
  } finally {
    predictButton.disabled = false;
    predictButton.textContent = "Analyze leaf";
  }
}

fileInput.addEventListener("change", () => {
  setPreview(fileInput.files[0]);
  setNeutralResult("Image ready. Click Analyze leaf to run the model.");
});

predictButton.addEventListener("click", analyzeLeaf);

["dragenter", "dragover"].forEach((eventName) => {
  dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropzone.classList.add("is-active");
  });
});

["dragleave", "drop"].forEach((eventName) => {
  dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropzone.classList.remove("is-active");
  });
});

dropzone.addEventListener("drop", (event) => {
  const [file] = event.dataTransfer.files;
  if (!file || !file.type.startsWith("image/")) {
    setNeutralResult("Please drop a valid image file.");
    return;
  }

  const dataTransfer = new DataTransfer();
  dataTransfer.items.add(file);
  fileInput.files = dataTransfer.files;
  setPreview(file);
  setNeutralResult("Image ready. Click Analyze leaf to run the model.");
});

document.querySelectorAll('input[name="modelChoice"]').forEach((input) => {
  input.addEventListener("change", () => {
    setNeutralResult("Model selected. Upload an image or click Analyze leaf again.");
  });
});

window.addEventListener("beforeunload", clearObjectUrl);
