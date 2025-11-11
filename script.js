const fileInput = document.querySelector('#video-upload');
const startProcessingButton = document.querySelector('#start-processing');
const clearSelectionButton = document.querySelector('#clear-selection');
const metadataName = document.querySelector('#metadata-name');
const metadataDuration = document.querySelector('#metadata-duration');
const metadataSize = document.querySelector('#metadata-size');
const metadataStatus = document.querySelector('#metadata-status');
const summaryStatus = document.querySelector('#summary-status');
const summaryPlaceholder = document.querySelector('#summary-placeholder');
const summaryContent = document.querySelector('#summary-content');
const summaryVideo = document.querySelector('#summary-video');
const summaryTextSection = document.querySelector('#summary-text');
const summaryTextContent = document.querySelector('#summary-text-content');
const highlightsList = document.querySelector('#highlights-list');
const toast = document.querySelector('#toast');
const downloadSummaryButton = document.querySelector('#download-summary');
const shareSummaryButton = document.querySelector('#share-summary');
const localModeIndicator = document.querySelector('#local-mode-indicator');

let uploadedFile = null;
let backendOnline = false;
let summaryObjectUrl = null;

const API_BASE_URL = 'http://localhost:8000';
const isDesktopShell = Boolean(window?.desktopApp?.isElectron);

const wait = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

const formatBytes = (bytes) => {
  if (!bytes && bytes !== 0) return '—';
  const thresholds = [
    { value: 1e9, suffix: 'GB' },
    { value: 1e6, suffix: 'MB' },
    { value: 1e3, suffix: 'KB' }
  ];

  for (const { value, suffix } of thresholds) {
    if (bytes >= value) {
      return `${(bytes / value).toFixed(1)} ${suffix}`;
    }
  }

  return `${bytes} B`;
};

const formatDuration = (seconds) => {
  if (!Number.isFinite(seconds)) return '—';
  const totalSeconds = Math.max(0, Math.round(seconds));
  const hrs = Math.floor(totalSeconds / 3600);
  const mins = Math.floor((totalSeconds % 3600) / 60);
  const secs = totalSeconds % 60;
  const hourPrefix = hrs ? `${hrs}:` : '';
  return `${hourPrefix}${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
};

const revokeSummaryUrl = () => {
  if (summaryObjectUrl) {
    URL.revokeObjectURL(summaryObjectUrl);
    summaryObjectUrl = null;
  }
};

const resetUI = () => {
  uploadedFile = null;
  revokeSummaryUrl();
  metadataName.textContent = '—';
  metadataDuration.textContent = '—';
  metadataSize.textContent = '—';
  metadataStatus.textContent = 'Waiting for upload';
  summaryStatus.textContent = 'No summary generated yet';
  startProcessingButton.disabled = true;
  clearSelectionButton.disabled = true;
  summaryContent.classList.add('hidden');
  summaryPlaceholder.classList.remove('hidden');
  summaryVideo.removeAttribute('src');
  if (summaryTextSection && summaryTextContent) {
    summaryTextSection.classList.add('hidden');
    summaryTextContent.textContent = '';
  }
  highlightsList.innerHTML = '';
};

const describeEnvironment = () => {
  if (!localModeIndicator) return;
  if (backendOnline) {
    localModeIndicator.textContent = isDesktopShell
      ? 'Secure desktop shell with backend connection'
      : 'Browser + backend ready';
  } else {
    localModeIndicator.textContent = isDesktopShell
      ? 'Desktop shell (backend offline, using mock mode)'
      : 'Browser only (backend offline, using mock mode)';
  }
};

const showToast = (message, duration = 4000) => {
  toast.textContent = message;
  toast.classList.add('visible');
  setTimeout(() => toast.classList.remove('visible'), duration);
};

const checkBackendHealth = async () => {
  try {
    const response = await fetch(`${API_BASE_URL}/`);
    backendOnline = response.ok;
    if (!backendOnline) throw new Error('Backend unavailable');
  } catch (error) {
    backendOnline = false;
  }
  describeEnvironment();
  return backendOnline;
};

const uploadVideoToBackend = async (file) => {
  const formData = new FormData();
  formData.append('file', file);

  const response = await fetch(`${API_BASE_URL}/upload`, {
    method: 'POST',
    body: formData
  });

  if (!response.ok) {
    const errorBody = await response.json().catch(() => ({}));
    throw new Error(errorBody.detail || 'Video upload failed');
  }

  return response.json();
};

const startBackendProcessing = async (fileId) => {
  const response = await fetch(`${API_BASE_URL}/process`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ file_id: fileId })
  });

  if (!response.ok) {
    const errorBody = await response.json().catch(() => ({}));
    throw new Error(errorBody.detail || 'Failed to start processing job');
  }

  return response.json();
};

const pollJobUntilComplete = async (jobId) => {
  let attempt = 0;
  while (true) {
    await wait(Math.min(1000 + attempt * 250, 3000));
    attempt += 1;

    const response = await fetch(`${API_BASE_URL}/jobs/${jobId}`);
    if (!response.ok) {
      const errorBody = await response.json().catch(() => ({}));
      throw new Error(errorBody.detail || 'Failed to fetch job status');
    }

    const job = await response.json();

    if (job.status === 'failed') {
      throw new Error(job.error || 'Processing failed');
    }

    if (job.status === 'completed') {
      const resultResponse = await fetch(`${API_BASE_URL}/jobs/${jobId}/result`);
      if (!resultResponse.ok) {
        const errorBody = await resultResponse.json().catch(() => ({}));
        throw new Error(errorBody.detail || 'Failed to fetch job result');
      }
      const { result } = await resultResponse.json();
      return result;
    }

    summaryStatus.textContent = `Processing... ${job.progress || 0}%`;
  }
};

const setVideoPreview = () => {
  if (!uploadedFile) return;
  revokeSummaryUrl();
  summaryObjectUrl = URL.createObjectURL(uploadedFile);
  summaryVideo.src = summaryObjectUrl;
  summaryVideo.load();
};

const createHighlightCard = (highlight, slice) => {
  const li = document.createElement('li');
  li.className = 'highlight-card';

  const preview = document.createElement('div');
  preview.className = 'highlight-preview';
  if (slice?.image_base64) {
    const img = document.createElement('img');
    img.src = `data:image/jpeg;base64,${slice.image_base64}`;
    img.alt = `Preview at ${highlight.timestamp}`;
    preview.appendChild(img);
  }

  const info = document.createElement('div');
  info.className = 'highlight-info';

  const meta = document.createElement('div');
  meta.className = 'highlight-meta';

  const timestamp = document.createElement('span');
  timestamp.className = 'timestamp';
  timestamp.textContent = highlight.timestamp;

  const label = document.createElement('span');
  label.className = 'label';
  label.textContent = highlight.label || 'Detected moment';

  meta.append(timestamp, label);

  const description = document.createElement('p');
  description.className = 'highlight-description';
  description.textContent = highlight.description || 'Awaiting detailed description from LLM.';

  const reviewButton = document.createElement('button');
  reviewButton.type = 'button';
  reviewButton.className = 'tertiary';
  reviewButton.textContent = 'Review';
  reviewButton.dataset.jump = highlight.time_seconds ?? 0;

  reviewButton.addEventListener('click', () => {
    if (typeof highlight.time_seconds === 'number') {
      summaryVideo.currentTime = highlight.time_seconds;
      summaryVideo.play();
    }
  });

  info.append(meta, description, reviewButton);
  li.append(preview, info);
  return li;
};

const renderBackendResult = (result) => {
  summaryContent.classList.remove('hidden');
  summaryPlaceholder.classList.add('hidden');

  setVideoPreview();

  if (summaryTextSection && summaryTextContent && result.summary_text) {
    summaryTextSection.classList.remove('hidden');
    summaryTextContent.textContent = result.summary_text;
  }

  highlightsList.innerHTML = '';
  const slicesByTimestamp = new Map();
  (result.frame_slices || []).forEach((slice) => {
    slicesByTimestamp.set(slice.timestamp, slice);
  });

  const highlights = result.highlights || [];
  if (!highlights.length && result.frame_slices?.length) {
    result.frame_slices.forEach((slice) => {
      highlights.push({
        timestamp: slice.timestamp,
        time_seconds: slice.time_seconds,
        label: 'Frame capture',
        description: 'Replace with LLM-generated insight.'
      });
    });
  }

  if (!highlights.length) {
    const emptyMessage = document.createElement('li');
    emptyMessage.textContent = 'No key moments detected in this stub summary.';
    highlightsList.appendChild(emptyMessage);
  } else {
    highlights.forEach((highlight) => {
      const slice = slicesByTimestamp.get(highlight.timestamp);
      const card = createHighlightCard(highlight, slice);
      highlightsList.appendChild(card);
    });
  }

  summaryStatus.textContent = 'Summary ready for review';
  metadataStatus.textContent = 'Complete';
  showToast('Summary complete. Review key moments and export when ready.');
};

const simulateSummary = () => {
  summaryContent.classList.remove('hidden');
  summaryPlaceholder.classList.add('hidden');
  summaryStatus.textContent = 'Summary ready for review (mock mode)';
  metadataStatus.textContent = 'Mock processing complete';
  setVideoPreview();

  if (summaryTextSection && summaryTextContent) {
    summaryTextSection.classList.remove('hidden');
    summaryTextContent.textContent = 'Backend offline. Displaying mock summary for UI verification.';
  }

  highlightsList.innerHTML = '';
  const mockHighlights = [
    { timestamp: '00:02:14', time_seconds: 134, label: 'Initial incision', description: 'Access established.' },
    { timestamp: '00:07:52', time_seconds: 472, label: 'Critical vessel secured', description: 'Critical vessel was clipped.' },
    { timestamp: '00:12:08', time_seconds: 728, label: 'Anastomosis check', description: 'Integrity confirmed.' }
  ];
  mockHighlights.forEach((highlight) => {
    const card = createHighlightCard(highlight, null);
    highlightsList.appendChild(card);
  });
};

fileInput.addEventListener('change', async (event) => {
  const [file] = event.target.files;
  if (!file) return;

  uploadedFile = file;
  metadataName.textContent = file.name;
  metadataSize.textContent = formatBytes(file.size);
  metadataStatus.textContent = 'Ready to process';
  startProcessingButton.disabled = false;
  clearSelectionButton.disabled = false;

  const fileURL = URL.createObjectURL(file);
  const tempVideo = document.createElement('video');
  tempVideo.preload = 'metadata';
  tempVideo.src = fileURL;
  tempVideo.addEventListener('loadedmetadata', () => {
    metadataDuration.textContent = formatDuration(tempVideo.duration);
    URL.revokeObjectURL(fileURL);
  });
});

startProcessingButton.addEventListener('click', async () => {
  if (!uploadedFile) return;

  startProcessingButton.disabled = true;
  clearSelectionButton.disabled = true;

  const backendHealthy = await checkBackendHealth();

  if (!backendHealthy) {
    showToast('Backend offline. Using mock summary for this session.', 5000);
    simulateSummary();
    startProcessingButton.disabled = false;
    clearSelectionButton.disabled = false;
    return;
  }

  try {
    summaryStatus.textContent = 'Uploading video...';
    metadataStatus.textContent = 'Uploading';

    const uploadResponse = await uploadVideoToBackend(uploadedFile);
    metadataStatus.textContent = 'Upload complete - queued for AI review';

    const { job_id: jobId } = await startBackendProcessing(uploadResponse.file_id);
    summaryStatus.textContent = 'Processing...';

    const result = await pollJobUntilComplete(jobId);
    renderBackendResult(result);
  } catch (error) {
    const message = error?.message || 'Processing failed';
    metadataStatus.textContent = 'Error during processing';
    summaryStatus.textContent = message;
    showToast(message, 6000);
  } finally {
    startProcessingButton.disabled = false;
    clearSelectionButton.disabled = false;
  }
});

clearSelectionButton.addEventListener('click', () => {
  fileInput.value = '';
  resetUI();
});

downloadSummaryButton.addEventListener('click', () => {
  showToast('Summary export queued. We will notify you when it is ready.');
});

shareSummaryButton.addEventListener('click', () => {
  showToast('Shared with the surgical review workspace.');
});

resetUI();
checkBackendHealth();
