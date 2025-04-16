// Wrap everything in a DOMContentLoaded listener to ensure HTML is loaded first
document.addEventListener('DOMContentLoaded', () => {
    'use strict'; // Enable strict mode for better error checking & performance

    // --- Element References ---
    const uploadForm = document.getElementById('upload-form');
    const fileInput = document.getElementById('file-input');
    const uploadStatus = document.getElementById('upload-status');
    const imageContainer = document.getElementById('image-container');
    const imageCanvas = document.getElementById('image-canvas');
    const ctx = imageCanvas.getContext('2d');
    const imageInfo = document.getElementById('image-info');
    const roiToggle = document.getElementById('roi-toggle');
    const roiSizeSlider = document.getElementById('roi-size-slider');
    const roiSizePx = document.getElementById('roi-size-px');
    const roiPhysicalSizeDiv = document.getElementById('roi-physical-size');
    const roiSizeMm = document.getElementById('roi-size-mm');
    const roiAreaMm2 = document.getElementById('roi-area-mm2');
    const roiXDisplay = document.getElementById('roi-x');
    const roiYDisplay = document.getElementById('roi-y');
    const gridRadios = document.querySelectorAll('input[name="grid"]');
    const recalculateButton = document.getElementById('recalculate-button');
    const centerRoiButton = document.getElementById('center-roi-button');
    const resultsDisplay = document.getElementById('results-display');
    const mainMeanDisplay = document.getElementById('main-mean');
    const subMeansGrid = document.getElementById('sub-means-grid');
    const roiErrorMessage = document.getElementById('error-message');
    const downloadButton = document.getElementById('download-button');
    const downloadStatus = document.getElementById('download-status');
    const sliceControlDiv = document.getElementById('slice-control');
    const sliceSlider = document.getElementById('slice-slider');
    const sliceInfo = document.getElementById('slice-info');
    const thresholdSection = document.getElementById('threshold-section');
    const enableThresholdingCheckbox = document.getElementById('enable-thresholding');
    const thresholdControlsContent = document.getElementById('threshold-controls-content');
    const imageRangeInfo = document.getElementById('image-range-info');
    const imgMinDisplay = document.getElementById('img-min');
    const imgMaxDisplay = document.getElementById('img-max');
    const thresholdListDiv = document.getElementById('threshold-list');
    const addThresholdButton = document.getElementById('add-threshold-button');
    const thresholdRowTemplate = document.getElementById('threshold-row-template');
    const exportColormapButton = document.getElementById('export-colormap-button');
    // const exportMaskButton = document.getElementById('export-mask-button'); // REMOVED
    const thresholdStatus = document.getElementById('threshold-status');
    const thresholdErrorMessage = document.getElementById('threshold-error-message');

    // --- State Variables ---
    let originalImage = null; let currentImageUrl = null; let baseImageUrl = null; let annotatedImageUrl = null; let thresholdImageUrl = null;
    let imageWidth = 0; let imageHeight = 0; let pix_dim_x = 1.0; let pix_dim_y = 1.0;
    let isNifti = false; let numSlices = 1; let currentSliceIndex = 0;
    let roi = { x: 50, y: 50, size: 100 }; let isRoiEnabled = false; let isDragging = false;
    let dragStartX_canvas, dragStartY_canvas; let roiStartX, roiStartY;
    let scaleX = 1; let scaleY = 1; let isDraggingRAFScheduled = false;
    let resizeTimeout; let wheelDebounceTimeout = null; const WHEEL_DEBOUNCE_MS = 50;
    const MOUSE_MOVE_THROTTLE_MS = 16; let lastMouseMoveTime = 0;
    let thresholds = []; let thresholdCounter = 0; let thresholdUpdateDebounce = null;
    const THRESHOLD_DEBOUNCE_MS = 300; let isThresholdingEnabled = false;

    // --- Initialization ---
    console.log("Initializing ROI Analyzer script...");
    clearStateAndUI();

    // --- Event Listeners Setup ---
    console.log("Setting up event listeners...");
    uploadForm.addEventListener('submit', handleImageUpload);
    roiToggle.addEventListener('change', handleRoiToggle);
    roiSizeSlider.addEventListener('input', handleRoiSizeChange);
    gridRadios.forEach(radio => radio.addEventListener('change', handleGridChange));
    imageCanvas.addEventListener('mousedown', handleRoiMouseDown);
    imageCanvas.addEventListener('mousemove', handleRoiMouseMove);
    imageCanvas.addEventListener('mouseup', handleRoiMouseUp);
    imageCanvas.addEventListener('mouseleave', handleRoiMouseLeave);
    centerRoiButton.addEventListener('click', () => triggerCalculation(true));
    recalculateButton.addEventListener('click', () => triggerCalculation(false));
    downloadButton.addEventListener('click', handleDownloadRequest);
    sliceSlider.addEventListener('input', handleSliceChange);
    imageCanvas.addEventListener('wheel', handleMouseWheelSliceChange, { passive: false });
    window.addEventListener('resize', () => { clearTimeout(resizeTimeout); resizeTimeout = setTimeout(handleWindowResize, 150); });
    enableThresholdingCheckbox.addEventListener('change', handleEnableThresholdingChange);
    addThresholdButton.addEventListener('click', addThresholdRow);
    thresholdListDiv.addEventListener('click', handleThresholdListClick);
    thresholdListDiv.addEventListener('input', handleThresholdInputChange);
    exportColormapButton.addEventListener('click', () => handleThresholdExport('colormap'));
    // exportMaskButton.addEventListener('click', () => handleThresholdExport('mask')); // REMOVED Listener
    console.log("Event listeners set up.");

    // --- Helper: Robust Fetch ---
    async function robustFetch(url, options = {}) {
        const response = await fetch(url, options);
        if (!response.ok) { let errorPayload; let errorText = `HTTP Error ${response.status}`; try { errorPayload = await response.json(); errorText = errorPayload?.error || JSON.stringify(errorPayload) || errorText; } catch (e) { try { errorText = await response.text(); } catch (et) {} } console.error("Fetch Error:", errorText); throw new Error(errorText); }
        try { const data = await response.json(); return data; }
        catch (e) { console.error("Fetch OK but not JSON:", e); throw new Error("Invalid response from server."); }
    }

    // --- Upload & Initial Setup ---
    async function handleImageUpload(event) { /* Keep as before */ event.preventDefault(); console.log("Upload triggered."); clearStateAndUI(); setUploadStatus('Uploading...', 'pending'); const file = fileInput.files[0]; if (!file) { showError('Select file.', roiErrorMessage); setUploadStatus(''); return; } const formData = new FormData(); formData.append('file', file); try { const result = await robustFetch('/upload', { method: 'POST', body: formData }); console.log("Upload successful:", result); setUploadStatus(`Success: ${result.filename}`, 'success'); baseImageUrl = result.image_url; currentImageUrl = baseImageUrl; imageWidth = result.image_width; imageHeight = result.image_height; pix_dim_x = result.pix_dim_x || 1.0; pix_dim_y = result.pix_dim_y || 1.0; isNifti = result.is_nifti || false; numSlices = result.num_slices || 1; currentSliceIndex = result.middle_slice_index || 0; imageInfo.textContent = `Dims: ${imageWidth}x${imageHeight} px`; updateRoiSizeDisplay(); setupNiftiUI(isNifti, numSlices, currentSliceIndex); loadImageToCanvas(currentImageUrl, () => { console.log("Initial load callback."); enableRoiControls(); roi.x = Math.max(0, Math.floor(imageWidth / 2 - roi.size / 2)); roi.y = Math.max(0, Math.floor(imageHeight / 2 - roi.size / 2)); updateRoiCoordsDisplay(); if(isRoiEnabled) drawClientRoiOverlay(); if (isNifti) fetchSliceInfo(currentSliceIndex); }); } catch (error) { console.error('Upload failed:', error); setUploadStatus('Upload failed.', 'error'); showError(`Upload Error: ${error.message}`, roiErrorMessage); clearStateAndUI(); } finally { fileInput.value = ''; } }
    function setupNiftiUI(isNiftiFile, sliceCount, middleIndex) { /* Keep as before */ renumberSections(); if (isNiftiFile) { console.log("Setting up NIfTI UI..."); thresholdSection.style.display = 'block'; enableThresholdingCheckbox.disabled = false; enableThresholdingCheckbox.checked = false; thresholdControlsContent.style.display = 'none'; setThresholdExportButtonsState(false); if (sliceCount > 1) { sliceSlider.max = sliceCount - 1; sliceSlider.value = middleIndex; sliceInfo.textContent = `Slice ${middleIndex + 1} / ${sliceCount}`; sliceControlDiv.style.display = 'flex'; sliceSlider.disabled = false; } else { sliceControlDiv.style.display = 'none'; sliceSlider.disabled = true; } } else { console.log("Setting up Standard Image UI..."); thresholdSection.style.display = 'none'; enableThresholdingCheckbox.disabled = true; enableThresholdingCheckbox.checked = false; thresholdControlsContent.style.display = 'none'; sliceControlDiv.style.display = 'none'; sliceSlider.disabled = true; } }
    async function fetchSliceInfo(sliceIdx) { /* Keep as before */ console.log(`Workspaceing info slice ${sliceIdx}`); imgMinDisplay.textContent="..."; imgMaxDisplay.textContent="..."; try { const data = await robustFetch(`/get_slice_info/${sliceIdx}`); imgMinDisplay.textContent = data.min_value?.toFixed(3) ?? 'N/A'; imgMaxDisplay.textContent = data.max_value?.toFixed(3) ?? 'N/A'; } catch (error) { imgMinDisplay.textContent = "Err"; imgMaxDisplay.textContent = "Err"; showError(`Slice Info Error: ${error.message}`, thresholdErrorMessage); } }

    // --- Slice & ROI Handling ---
    async function handleSliceChange() { /* Keep as before */ const newSliceIndex = parseInt(sliceSlider.value, 10); if (newSliceIndex === currentSliceIndex) return; currentSliceIndex = newSliceIndex; sliceInfo.textContent = `Slice ${currentSliceIndex + 1} / ${numSlices}`; console.log(`Slice changed to index: ${currentSliceIndex}`); annotatedImageUrl = null; thresholdImageUrl = null; clearRoiResults(); showError('', roiErrorMessage); showError('', thresholdErrorMessage); setThresholdStatus(''); fetchSliceInfo(currentSliceIndex); if (isThresholdingEnabled && thresholds.length > 0) { updateThresholdedDisplay(); } else { setUploadStatus(`Loading slice ${currentSliceIndex + 1}...`, 'pending'); try { const data = await robustFetch(`/get_slice/${currentSliceIndex}`); baseImageUrl = data.image_url; currentImageUrl = baseImageUrl; loadImageToCanvas(currentImageUrl, () => { setUploadStatus(''); if (isRoiEnabled) drawClientRoiOverlay(); }); } catch (error) { showError(`Slice Load Error: ${error.message}`, roiErrorMessage); setUploadStatus('Slice error.', 'error'); } } }
    function handleMouseWheelSliceChange(event) { /* Keep as before */ if (!isNifti || numSlices <= 1 || isDragging) return; event.preventDefault(); clearTimeout(wheelDebounceTimeout); wheelDebounceTimeout = setTimeout(() => { const dir = event.deltaY > 0 ? 1 : -1; let newIdx = currentSliceIndex + dir; newIdx = Math.max(0, Math.min(newIdx, numSlices - 1)); if (newIdx !== currentSliceIndex) { sliceSlider.value = newIdx; handleSliceChange(); } }, WHEEL_DEBOUNCE_MS); }
    function handleRoiToggle() { /* Keep as before */ isRoiEnabled = roiToggle.checked; console.log(`ROI Enabled: ${isRoiEnabled}`); annotatedImageUrl = null; currentImageUrl = thresholdImageUrl || baseImageUrl; setRoiControlsState(isRoiEnabled); if (originalImage) { loadImageToCanvas(currentImageUrl, () => { if(isRoiEnabled) drawClientRoiOverlay(); }); } else { updateRoiCoordsDisplay(); } }
    function handleRoiSizeChange() { /* Keep as before */ roi.size = parseInt(roiSizeSlider.value, 10); console.log(`ROI size: ${roi.size} px`); updateRoiSizeDisplay(); if (originalImage && isRoiEnabled) { roi.x = Math.max(0, Math.min(roi.x, imageWidth - roi.size)); roi.y = Math.max(0, Math.min(roi.y, imageHeight - roi.size)); updateRoiCoordsDisplay(); currentImageUrl = thresholdImageUrl || baseImageUrl; annotatedImageUrl = null; loadImageToCanvas(currentImageUrl, drawClientRoiOverlay); setCalculationButtonsState(true); } }
    function handleGridChange() { /* Keep as before */ console.log(`Grid: ${document.querySelector('input[name="grid"]:checked').value}`); if (isRoiEnabled && originalImage) { currentImageUrl = thresholdImageUrl || baseImageUrl; annotatedImageUrl = null; loadImageToCanvas(currentImageUrl, drawClientRoiOverlay); setCalculationButtonsState(true); } }
    function handleRoiMouseDown(event) { /* Keep as before */ if (!isRoiEnabled || !originalImage || isDragging) return; const mousePos = getMousePosOnCanvas(event); const roiDisplay = getRoiDisplayCoords(); if (mousePos.x >= roiDisplay.x && mousePos.x <= roiDisplay.x + roiDisplay.width && mousePos.y >= roiDisplay.y && mousePos.y <= roiDisplay.y + roiDisplay.height) { console.log("Start drag."); isDragging = true; annotatedImageUrl = null; dragStartX_canvas = mousePos.x; dragStartY_canvas = mousePos.y; roiStartX = roi.x; roiStartY = roi.y; imageCanvas.style.cursor = 'grabbing'; } }
    function handleRoiMouseMove(event) { /* Keep as before */ if (!isDragging || !isRoiEnabled || !originalImage) return; const now = performance.now(); if (now - lastMouseMoveTime < MOUSE_MOVE_THROTTLE_MS) return; lastMouseMoveTime = now; const mousePos = getMousePosOnCanvas(event); const deltaX_img = scaleX !== 0 ? (mousePos.x - dragStartX_canvas) / scaleX : 0; const deltaY_img = scaleY !== 0 ? (mousePos.y - dragStartY_canvas) / scaleY : 0; let newRoiX = Math.round(roiStartX + deltaX_img); let newRoiY = Math.round(roiStartY + deltaY_img); newRoiX = Math.max(0, Math.min(newRoiX, imageWidth - roi.size)); newRoiY = Math.max(0, Math.min(newRoiY, imageHeight - roi.size)); if (newRoiX !== roi.x || newRoiY !== roi.y) { roi.x = newRoiX; roi.y = newRoiY; updateRoiCoordsDisplay(); if (!isDraggingRAFScheduled) { isDraggingRAFScheduled = true; window.requestAnimationFrame(() => { if (isDragging) drawClientRoiOverlay(); isDraggingRAFScheduled = false; }); } } }
    function handleRoiMouseUp() { stopDragging(); }
    function handleRoiMouseLeave() { stopDragging(); }
    function stopDragging() { /* Keep as before */ if (isDragging) { console.log("Stop drag."); isDragging = false; imageCanvas.style.cursor = 'crosshair'; isDraggingRAFScheduled = false; if (isRoiEnabled) { setCalculationButtonsState(true); console.log("Final draw."); loadImageToCanvas(currentImageUrl, drawClientRoiOverlay); } } }
    async function triggerCalculation(centerOnMinIntensity = false) { /* Keep as before */ if (!isRoiEnabled || !originalImage) { showError('Enable ROI.', roiErrorMessage); return; } console.log(`Triggering calculation: center=${centerOnMinIntensity}, slice=${currentSliceIndex}`); clearRoiResults(); showError('', roiErrorMessage); showError('', thresholdErrorMessage); setCalculationButtonsState(false); downloadButton.disabled = true; annotatedImageUrl = null; thresholdImageUrl = null; const selectedGrid = document.querySelector('input[name="grid"]:checked').value; try { const result = await robustFetch('/calculate_roi', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ x: roi.x, y: roi.y, size: roi.size, grid: parseInt(selectedGrid, 10), center_on_min: centerOnMinIntensity, slice_index: isNifti ? currentSliceIndex : null }) }); console.log("Calculation response:", result); if (result.final_roi_x !== undefined) roi.x = result.final_roi_x; if (result.final_roi_y !== undefined) roi.y = result.final_roi_y; updateRoiCoordsDisplay(); const resultsData = result.results; if (!resultsData) throw new Error("Invalid response."); mainMeanDisplay.textContent = resultsData.main_mean ?? 'N/A'; subMeansGrid.innerHTML = ''; if (resultsData.means?.length > 0) { const gridDim = Math.round(Math.sqrt(resultsData.means.length)); subMeansGrid.style.gridTemplateColumns = `repeat(${Math.min(gridDim || 1, 5)}, 1fr)`; resultsData.means.forEach(mean => { const span = document.createElement('span'); span.textContent = mean ?? 'N/A'; subMeansGrid.appendChild(span); }); } else { subMeansGrid.textContent = (parseInt(selectedGrid, 10) > 1) ? '(No sub-ROIs)' : '(1x1)'; } if (result.annotated_image) { annotatedImageUrl = result.annotated_image; currentImageUrl = annotatedImageUrl; loadImageToCanvas(annotatedImageUrl, () => { console.log("Annotated image displayed."); updateRoiCoordsDisplay(); }); } else { currentImageUrl = baseImageUrl; loadImageToCanvas(currentImageUrl, drawClientRoiOverlay); } downloadButton.disabled = false; } catch (error) { console.error('Calculation failed:', error); showError(`Calculation Error: ${error.message}`, roiErrorMessage); clearRoiResults(); downloadButton.disabled = true; annotatedImageUrl = null; currentImageUrl = baseImageUrl; loadImageToCanvas(currentImageUrl, drawClientRoiOverlay); } finally { setCalculationButtonsState(isRoiEnabled); } }
    async function handleDownloadRequest() { /* Keep robust version */ console.log("DL Excel triggered."); setDownloadStatus('Preparing...', 'pending'); showError('', roiErrorMessage); try { const response = await fetch('/download_excel'); if (!response.ok) { const txt = await response.text(); throw new Error(txt || `Download failed`); } const disposition = response.headers.get('Content-Disposition'); let filename = 'roi_means.xlsx'; if (disposition?.includes('attachment')) { const m = disposition.match(/filename\*?=['"]?([^'";]+)['"]?/); if (m?.[1]) filename = decodeURIComponent(m[1]); } const blob = await response.blob(); const url = window.URL.createObjectURL(blob); const a = document.createElement('a'); a.style.display = 'none'; a.href = url; a.download = filename; document.body.appendChild(a); a.click(); document.body.removeChild(a); window.URL.revokeObjectURL(url); setDownloadStatus('Download started.', 'success'); setTimeout(() => { setDownloadStatus(''); }, 5000); } catch (error) { console.error('DL Excel failed:', error); setDownloadStatus('Download failed.', 'error'); showError(`DL Error: ${error.message}`, roiErrorMessage); } }
    function handleWindowResize() { /* Keep robust version */ console.log("Debounced resize."); if (originalImage) { loadImageToCanvas(currentImageUrl, () => { if (isRoiEnabled && !isDragging) drawClientRoiOverlay(); updateRoiCoordsDisplay(); }); } }

    // --- Thresholding Functions ---

    /** Handles the Enable Thresholding checkbox change */
    function handleEnableThresholdingChange(event) {
        isThresholdingEnabled = event.target.checked;
        console.log("Thresholding Enabled:", isThresholdingEnabled);
        thresholdControlsContent.style.display = isThresholdingEnabled ? 'block' : 'none';
        showError('', thresholdErrorMessage); setThresholdStatus('');

        if (isThresholdingEnabled) {
            annotatedImageUrl = null; // ROI annotations are hidden when thresholding is active
            if (thresholds.length === 0) {
                applyDefaultThresholds(); // Apply defaults if list is empty
            } else {
                updateThresholdedDisplay(); // Apply existing thresholds immediately
            }
            setThresholdExportButtonsState(thresholds.length > 0);
        } else {
            // Disable thresholding: revert view
            clearThresholds(); // Clear state array and UI rows
            thresholdImageUrl = null; // Clear threshold view state
            currentImageUrl = annotatedImageUrl || baseImageUrl; // Show ROI annotation or base
            loadImageToCanvas(currentImageUrl, () => { if (isRoiEnabled) drawClientRoiOverlay(); });
            setThresholdExportButtonsState(false);
        }
    }

    /** Applies a predefined set of default thresholds */
    function applyDefaultThresholds() {
        console.log("Applying default thresholds.");
        clearThresholds(); // Clear existing state and UI rows first
        const defaults = [ // Value, ColorHex
            [100, '#0000FF'], [400, '#FFFF00'], [800, '#00FF00'], [1000, '#FF0000']
        ];
        defaults.forEach(([value, color]) => addThresholdRowUI(value, color)); // Add rows to UI
        console.log("Default thresholds applied state:", thresholds);
        setThresholdExportButtonsState(true);
        updateThresholdedDisplay(); // Trigger single display update
    }

    /** Adds a threshold row to UI AND state */
    function addThresholdRow(initialValue = null, initialColor = null) {
        addThresholdRowUI(initialValue, initialColor); // Add to UI and state
        if (isThresholdingEnabled) updateThresholdedDisplay(); // Update display if active
    }

    /** Helper function to add row to UI and state (used by addThresholdRow and applyDefaultThresholds) */
    function addThresholdRowUI(initialValue = null, initialColor = null) {
        thresholdCounter++; const uniqueId = `threshold-${thresholdCounter}`;
        const template = thresholdRowTemplate.content.cloneNode(true);
        const rowDiv = template.querySelector('.threshold-row');
        const valueInput = template.querySelector('.threshold-value');
        const colorInput = template.querySelector('.threshold-color');
        if (!rowDiv || !valueInput || !colorInput) { console.error("Template error."); return; }
        rowDiv.dataset.id = uniqueId;
        let value = initialValue;
        if (value === null) {
            const sorted = [...thresholds].sort((a, b) => a.value - b.value);
            value = sorted.length > 0 ? sorted[sorted.length - 1].value + 100 : (parseFloat(imgMinDisplay.textContent) || 0);
        }
        const color = initialColor ?? '#FFA500';
        valueInput.value = value; colorInput.value = color;
        thresholdListDiv.appendChild(template);
        thresholds.push({ id: uniqueId, value: value, color: color });
    }

    /** Handles clicks within the threshold list (remove buttons) */
    function handleThresholdListClick(event) { /* Keep previous version */ if (event.target.classList.contains('remove-threshold-button')) { const row = event.target.closest('.threshold-row'); if (row) { const id = row.dataset.id; row.remove(); thresholds = thresholds.filter(t => t.id !== id); console.log("Removed threshold:", id); setThresholdExportButtonsState(thresholds.length > 0); if (isThresholdingEnabled) updateThresholdedDisplay(); } } }
    /** Handles input changes in threshold fields (debounced) */
    function handleThresholdInputChange(event) { /* Keep previous version */ const target = event.target; const row = target.closest('.threshold-row'); if (!row) return; const id = row.dataset.id; const threshold = thresholds.find(t => t.id === id); if (!threshold) return; if (target.classList.contains('threshold-value')) threshold.value = parseFloat(target.value) || 0; else if (target.classList.contains('threshold-color')) threshold.color = target.value; clearTimeout(thresholdUpdateDebounce); thresholdUpdateDebounce = setTimeout(() => { console.log("Debounced change."); if (isThresholdingEnabled) updateThresholdedDisplay(); }, THRESHOLD_DEBOUNCE_MS); }

    /** Sends current threshold data to backend to get color-mapped image */
    async function updateThresholdedDisplay() {
        if (!isNifti || !isThresholdingEnabled) { if (thresholdImageUrl) { console.log("Reverting from threshold view."); currentImageUrl = annotatedImageUrl || baseImageUrl; thresholdImageUrl = null; loadImageToCanvas(currentImageUrl, () => { if(isRoiEnabled) drawClientRoiOverlay(); }); } return; }
        if (thresholds.length === 0) { console.log("Thresholding enabled but no thresholds."); currentImageUrl = baseImageUrl; thresholdImageUrl = null; annotatedImageUrl = null; loadImageToCanvas(currentImageUrl, () => { if(isRoiEnabled) drawClientRoiOverlay(); }); setThresholdExportButtonsState(false); setThresholdStatus('Add thresholds.'); return; }
        console.log("Updating thresholded display..."); setThresholdStatus('Applying...', 'pending'); showError('', thresholdErrorMessage);
        const sortedThresholds = [...thresholds].sort((a, b) => a.value - b.value);
        const payload = { slice_index: currentSliceIndex, thresholds: sortedThresholds.map(t => ({ value: t.value, color: t.color })) };
        try {
            const result = await robustFetch('/get_thresholded_slice', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
            thresholdImageUrl = result.image_url; currentImageUrl = thresholdImageUrl; annotatedImageUrl = null;
            loadImageToCanvas(currentImageUrl, () => { setThresholdStatus('Map updated.', 'success'); }); setThresholdExportButtonsState(true);
        } catch (error) { console.error("Error updating threshold display:", error); showError(`Threshold Error: ${error.message}`, thresholdErrorMessage); setThresholdStatus('Error.', 'error'); currentImageUrl = baseImageUrl; thresholdImageUrl = null; annotatedImageUrl = null; loadImageToCanvas(currentImageUrl, () => { if(isRoiEnabled) drawClientRoiOverlay(); }); setThresholdExportButtonsState(false); }
    }

    /** Handles export requests for colormap PNG (Mask export removed) */
    async function handleThresholdExport(exportType) { // 'colormap' ONLY now
        if (exportType !== 'colormap') { console.warn("Mask export removed."); return; } // Exit if mask requested
        if (!isNifti || thresholds.length === 0 || !isThresholdingEnabled) { showError(`Export failed: Thresholding not active or no thresholds.`, thresholdErrorMessage); return; }
        console.log(`Export request: ${exportType}`); setThresholdStatus(`Generating ${exportType}...`, 'pending'); showError('', thresholdErrorMessage);
        const backendUrl = '/download_colormap_png'; // Only colormap URL needed
        const expectedExtension = '.png';
        const sortedThresholds = [...thresholds].sort((a, b) => a.value - b.value);
        const payload = { slice_index: currentSliceIndex, thresholds: sortedThresholds.map(t => ({ value: t.value, color: t.color })) };
        try {
            const response = await fetch(backendUrl, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
            if (!response.ok) { const txt = await response.text(); throw new Error(txt || `Export failed`); }
            const disposition = response.headers.get('Content-Disposition'); let filename = `${exportType}${expectedExtension}`;
            if (disposition?.includes('attachment')) { const m = disposition.match(/filename\*?=['"]?([^'";]+)['"]?/); if (m?.[1]) { filename = decodeURIComponent(m[1]); if (!filename.toLowerCase().endsWith(expectedExtension)) filename += expectedExtension; } }
            console.log(`Download filename: ${filename}`); const blob = await response.blob(); const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a'); a.style.display = 'none'; a.href = url; a.download = filename;
            document.body.appendChild(a); a.click(); document.body.removeChild(a); window.URL.revokeObjectURL(url);
            console.log(`${exportType} download initiated.`); setThresholdStatus(`${exportType} download started.`, 'success'); setTimeout(() => { setThresholdStatus(''); }, 5000);
        } catch (error) { console.error(`Export failed (${exportType}):`, error); setThresholdStatus(`Export failed.`, 'error'); showError(`Export Error: ${error.message}`, thresholdErrorMessage); }
    }

    // --- UI Update and Helper Functions ---
    /** Loads an image (from URL or DataURL) onto the canvas */
    function loadImageToCanvas(imageUrl, callback) { /* Keep robust version */ if (!imageUrl) { console.error("loadImageToCanvas: null URL."); return; } console.log("loadImageToCanvas starting:", imageUrl.substring(0, 60) + "..."); const img = new Image(); img.onload = () => { console.log(`>>> Image loaded: ${img.width}x${img.height}`); originalImage = img; const containerWidth = imageContainer.clientWidth; const containerHeight = imageContainer.clientHeight; const imgAspectRatio = originalImage.width / originalImage.height; const containerAspectRatio = containerWidth > 0 && containerHeight > 0 ? containerWidth / containerHeight : imgAspectRatio; let displayWidth, displayHeight; if (imgAspectRatio > containerAspectRatio) { displayWidth = containerWidth; displayHeight = displayWidth / imgAspectRatio; } else { displayHeight = containerHeight; displayWidth = displayHeight * imgAspectRatio; } imageCanvas.width = Math.max(1, Math.round(displayWidth)); imageCanvas.height = Math.max(1, Math.round(displayHeight)); scaleX = (originalImage.width > 0) ? imageCanvas.width / originalImage.width : 1; scaleY = (originalImage.height > 0) ? imageCanvas.height / originalImage.height : 1; console.log(`Canvas: ${imageCanvas.width}x${imageCanvas.height}, Scale: ${scaleX.toFixed(3)}x${scaleY.toFixed(3)}`); try { ctx.clearRect(0, 0, imageCanvas.width, imageCanvas.height); ctx.drawImage(originalImage, 0, 0, imageCanvas.width, imageCanvas.height); console.log(">>> Image drawn."); } catch (e) { console.error("drawImage error:", e); showError("Error drawing image.", roiErrorMessage); return; } if (callback) callback(); }; img.onerror = (e) => { console.error("Image load error:", e); showError('Error loading image source.', roiErrorMessage); clearStateAndUI(); }; img.src = imageUrl + (imageUrl.startsWith('data:') ? '' : '?t=' + Date.now()); }
    /** Draws the client-side ROI box and grid lines overlay */
    function drawClientRoiOverlay() { if (!originalImage) return; if (!isDragging && !annotatedImageUrl && !thresholdImageUrl) { ctx.clearRect(0, 0, imageCanvas.width, imageCanvas.height); ctx.drawImage(originalImage, 0, 0, imageCanvas.width, imageCanvas.height); } else if (isDragging) { ctx.clearRect(0, 0, imageCanvas.width, imageCanvas.height); ctx.drawImage(originalImage, 0, 0, imageCanvas.width, imageCanvas.height); } if (!isRoiEnabled || annotatedImageUrl || thresholdImageUrl) { return; } const roiDisplay = getRoiDisplayCoords(); if (roiDisplay.width <= 0 || roiDisplay.height <= 0) return; ctx.strokeStyle = 'red'; ctx.lineWidth = 2; ctx.strokeRect(roiDisplay.x, roiDisplay.y, roiDisplay.width, roiDisplay.height); const grid = parseInt(document.querySelector('input[name="grid"]:checked').value, 10); if (grid > 1) { const subX = roiDisplay.width / grid; const subY = roiDisplay.height / grid; if (subX >= 1 && subY >= 1) { ctx.save(); ctx.strokeStyle = 'rgba(0, 0, 255, 0.7)'; ctx.lineWidth = 1; ctx.setLineDash([4, 3]); for (let i = 1; i < grid; i++) { const x = roiDisplay.x + i * subX; ctx.beginPath(); ctx.moveTo(x, roiDisplay.y); ctx.lineTo(x, roiDisplay.y + roiDisplay.height); ctx.stroke(); } for (let i = 1; i < grid; i++) { const y = roiDisplay.y + i * subY; ctx.beginPath(); ctx.moveTo(roiDisplay.x, y); ctx.lineTo(roiDisplay.x + roiDisplay.width, y); ctx.stroke(); } ctx.restore(); } } }
    /** Calculates ROI display coordinates */
    function getRoiDisplayCoords() { return { x: roi.x * scaleX, y: roi.y * scaleY, width: roi.size * scaleX, height: roi.size * scaleY }; }
    /** Calculates mouse position relative to canvas */
    function getMousePosOnCanvas(event) { const rect = imageCanvas.getBoundingClientRect(); return { x: event.clientX - rect.left, y: event.clientY - rect.top }; }
    /** Updates displayed ROI size */
    function updateRoiSizeDisplay() { const sPx=Number(roi.size)||0; roiSizePx.textContent=`${sPx} px`; if(pix_dim_x>0&&pix_dim_y>0&&(pix_dim_x!==1.0||pix_dim_y!==1.0)){const sx=(sPx*pix_dim_x); const sy=(sPx*pix_dim_y); const area=sx*sy; roiSizeMm.textContent=`${sx.toFixed(2)}x${sy.toFixed(2)}`; roiAreaMm2.textContent=area.toFixed(2); roiPhysicalSizeDiv.style.display='block';}else{roiPhysicalSizeDiv.style.display='none';} }
    /** Updates displayed ROI coordinates */
    function updateRoiCoordsDisplay() { roiXDisplay.textContent = (isRoiEnabled && originalImage) ? roi.x : 'N/A'; roiYDisplay.textContent = (isRoiEnabled && originalImage) ? roi.y : 'N/A'; }
    /** Enables base ROI controls */
    function enableRoiControls() { console.log("Enabling ROI controls."); roiToggle.disabled = false; const w = Number(imageWidth)||500; const h = Number(imageHeight)||500; const maxR = Math.min(w, h); roiSizeSlider.max = maxR >= 10 ? maxR : 10; roi.size = Math.max(10, Math.min(roi.size, parseInt(roiSizeSlider.max))); roiSizeSlider.value = roi.size; updateRoiSizeDisplay(); setRoiControlsState(roiToggle.checked); }
    /** Sets enabled/disabled state of ROI controls */
    function setRoiControlsState(enabled) { console.log(`Setting ROI controls state: ${enabled}`); roiSizeSlider.disabled = !enabled; gridRadios.forEach(radio => radio.disabled = !enabled); setCalculationButtonsState(enabled && originalImage); updateRoiCoordsDisplay(); }
    /** Sets enabled/disabled state of calculation buttons */
    function setCalculationButtonsState(enabled) { recalculateButton.disabled = !enabled; centerRoiButton.disabled = !enabled; }
    /** Sets enabled/disabled state of threshold export buttons */
    function setThresholdExportButtonsState(enabled) { exportColormapButton.disabled = !enabled; /* exportMaskButton.disabled = !enabled; */ } // Removed mask button disable
    /** Clears state and resets UI */
    function clearStateAndUI() { /* Keep previous version, ensure slice/threshold controls reset */ console.log("Clearing state and UI."); try { ctx.clearRect(0, 0, imageCanvas.width, imageCanvas.height); } catch(e){} imageCanvas.width = 300; imageCanvas.height = 200; imageInfo.textContent = 'No image loaded.'; setUploadStatus(''); setDownloadStatus(''); showError('', roiErrorMessage); showError('', thresholdErrorMessage); mainMeanDisplay.textContent = 'N/A'; subMeansGrid.innerHTML = 'N/A'; subMeansGrid.style.gridTemplateColumns = ''; originalImage = null; currentImageUrl = null; baseImageUrl = null; annotatedImageUrl = null; thresholdImageUrl = null; imageWidth = 0; imageHeight = 0; pix_dim_x = 1.0; pix_dim_y = 1.0; isNifti = false; numSlices = 1; currentSliceIndex = 0; roi = { x: 50, y: 50, size: 100 }; isDragging = false; scaleX = 1; scaleY = 1; roiToggle.checked = false; isRoiEnabled = false; roiToggle.disabled = true; roiSizeSlider.value = 100; roiSizeSlider.max = 500; updateRoiSizeDisplay(); updateRoiCoordsDisplay(); roiPhysicalSizeDiv.style.display = 'none'; gridRadios.forEach(radio => { radio.checked = (radio.value === '1'); }); setRoiControlsState(false); downloadButton.disabled = true; sliceControlDiv.style.display = 'none'; sliceSlider.disabled = true; sliceSlider.value=0; sliceSlider.max=1; sliceInfo.textContent='Slice 1 / 1'; thresholdSection.style.display = 'none'; thresholdListDiv.innerHTML = ''; thresholds = []; thresholdCounter = 0; imgMinDisplay.textContent = 'N/A'; imgMaxDisplay.textContent = 'N/A'; setThresholdExportButtonsState(false); setThresholdStatus(''); enableThresholdingCheckbox.checked = false; enableThresholdingCheckbox.disabled = true; thresholdControlsContent.style.display = 'none'; isThresholdingEnabled = false; renumberSections();}
    /** Displays an error message in a specific element */
    function showError(message, element = roiErrorMessage) { if(element){ element.textContent = message; element.style.display = message ? 'block' : 'none'; } }
    /** Clears ROI results display */
    function clearRoiResults() { mainMeanDisplay.textContent = 'N/A'; subMeansGrid.innerHTML = 'N/A'; subMeansGrid.style.gridTemplateColumns = ''; }
    /** Sets upload status message */
    function setUploadStatus(message, type = '') { uploadStatus.textContent = message; uploadStatus.className = type; }
    /** Sets download status message */
    function setDownloadStatus(message, type = '') { downloadStatus.textContent = message; downloadStatus.className = type; }
    /** Sets threshold status message */
    function setThresholdStatus(message, type = '') { thresholdStatus.textContent = message; thresholdStatus.className = type; }
    /** Clears threshold state array and UI list */
    function clearThresholds() { thresholds = []; thresholdListDiv.innerHTML = ''; thresholdCounter = 0; console.log("Thresholds cleared."); }
    /** Renumbers section headings based on NIfTI status */
    function renumberSections() { try{ let num = 3; document.querySelector('.results-section h2').firstChild.textContent = `${num++}. `; document.querySelector('.export-section h2').firstChild.textContent = `${num++}. `; if(isNifti) { document.querySelector('.threshold-section h2').firstChild.textContent = `${num++}. `;} console.log("Sections renumbered."); } catch(e){ console.warn("Renumbering failed - sections not found?"); } }

}); // End DOMContentLoaded Wrapper