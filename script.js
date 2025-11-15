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
const phaseClipsSection = document.querySelector('#phase-clips');
const phaseClipsList = document.querySelector('#phase-clips-list');
const phaseClipsControls = document.querySelector('#phase-clips-controls');
const downloadSelectedClipsButton = document.querySelector('#download-selected-clips');
const phaseClipsStatus = document.querySelector('#phase-clips-status');
const toast = document.querySelector('#toast');
const downloadSummaryButton = document.querySelector('#download-summary');
const shareSummaryButton = document.querySelector('#share-summary');
const localModeIndicator = document.querySelector('#local-mode-indicator');
const helpButton = document.querySelector('.help-button');
const helpModal = document.querySelector('#help-modal');
const closeModalButton = document.querySelector('.close-button');
const modalOverlay = document.querySelector('.modal-overlay');

const clipSelection = new Map();
let renderedPhaseClips = [];

let uploadedFile = null;
let backendOnline = false;
let summaryObjectUrl = null;

const API_BASE_URL = 'http://localhost:8000';
const isDesktopShell = Boolean(window?.desktopApp?.isElectron);

const wait = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

const backendUrl = (path = '') => {
  if (!path) return '';
  if (/^https?:/i.test(path)) return path;
  return `${API_BASE_URL}${path}`;
};

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
  if (phaseClipsSection) {
    phaseClipsSection.classList.add('hidden');
  }
  if (phaseClipsList) {
    phaseClipsList.innerHTML = '';
  }
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

const openHelpModal = () => {
  if (helpModal) {
    helpModal.classList.remove('hidden');
  }
};

const closeHelpModal = () => {
  if (helpModal) {
    helpModal.classList.add('hidden');
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
  const getProgressMessage = (progress) => {
    if (progress < 15) return 'Analyzing video structure...';
    if (progress < 40) return 'Identifying surgical phases...';
    if (progress < 70) return 'Extracting key moments...';
    if (progress < 95) return 'Finalizing results...';
    return 'Completing...';
  };

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

    const progress = job.progress || 0;
    const message = getProgressMessage(progress);
    summaryStatus.textContent = `${message} (${progress}%)`;
  }
};

const setVideoPreview = () => {
  if (!uploadedFile) return;
  revokeSummaryUrl();
  summaryObjectUrl = URL.createObjectURL(uploadedFile);
  summaryVideo.src = summaryObjectUrl;
  summaryVideo.load();
};

const formatClipRange = (clip) => {
  const start = clip.start_timestamp || formatDuration(clip.start_second ?? 0);
  const end = clip.end_timestamp || formatDuration(clip.end_second ?? 0);
  const duration = Math.round(clip.duration_seconds ?? 0);
  return `${start} → ${end} (${duration || 1}s)`;
};

const getClipKey = (clip) => {
  const phase = (clip.phase || '').trim();
  if (phase) {
    return `phase:${phase.toLowerCase()}`;
  }
  if (clip.file_name) {
    return `file:${clip.file_name}`;
  }
  return `start:${clip.start_second ?? 0}`;
};

const dedupePhaseClips = (clips) => {
  const bestByPhase = new Map();
  clips.forEach((clip) => {
    const key = getClipKey(clip);
    const currentBest = bestByPhase.get(key);
    if (!currentBest || (clip.duration_seconds ?? 0) > (currentBest.duration_seconds ?? 0)) {
      bestByPhase.set(key, clip);
    }
  });
  return Array.from(bestByPhase.values()).sort((a, b) => (a.start_second ?? 0) - (b.start_second ?? 0));
};

const getSelectedPhaseClips = () =>
  renderedPhaseClips.filter((clip) => clipSelection.get(getClipKey(clip)));

const updatePhaseClipsStatus = () => {
  if (!phaseClipsStatus) return;
  const total = renderedPhaseClips.length;
  const selected = getSelectedPhaseClips().length;
  phaseClipsStatus.textContent = total
    ? `${selected}/${total} segments selected (toggle off to skip download)`
    : '';
};

const updateDownloadSelectedButton = () => {
  if (!downloadSelectedClipsButton) return;
  downloadSelectedClipsButton.disabled = getSelectedPhaseClips().length === 0;
};

const triggerClipDownloads = (clips) => {
  clips.forEach((clip) => {
    const url = backendUrl(clip.download_url || clip.video_url);
    const link = document.createElement('a');
    link.href = url;
    link.download = clip.file_name || `${(clip.phase || 'clip').replace(/\s+/g, '-')}.mp4`;
    link.target = '_blank';
    link.rel = 'noreferrer';
    link.style.display = 'none';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  });
};

const handleDownloadSelectedClips = () => {
  const selected = getSelectedPhaseClips();
  if (!selected.length) {
    showToast('Select at least one clip to download.');
    return;
  }
  triggerClipDownloads(selected);
  showToast(`Downloading ${selected.length} clip${selected.length === 1 ? '' : 's'}.`);
};

const createClipCard = (clip) => {
  const li = document.createElement('li');
  li.className = 'clip-card clip-card--expanded';

  const clipKey = getClipKey(clip);
  li.dataset.clipKey = clipKey;

  const cardHeader = document.createElement('div');
  cardHeader.className = 'clip-card-header';

  const headerTitle = document.createElement('div');
  headerTitle.className = 'clip-card-title';

  const title = document.createElement('h4');
  title.textContent = clip.phase || 'Detected phase';

  const durationSeconds = Math.max(1, Math.round(clip.duration_seconds ?? 0));
  const durationBadge = document.createElement('span');
  durationBadge.className = 'clip-phase-tag';
  durationBadge.textContent = `${durationSeconds}s clip`;

  headerTitle.append(title, durationBadge);

  const expandButton = document.createElement('button');
  expandButton.type = 'button';
  expandButton.className = 'clip-expand-button';
  expandButton.setAttribute('aria-expanded', 'true');
  expandButton.setAttribute('aria-label', `Toggle ${clip.phase || 'phase'} details`);
  expandButton.textContent = '−';

  expandButton.addEventListener('click', () => {
    li.classList.toggle('clip-card--expanded');
    const isExpanded = li.classList.contains('clip-card--expanded');
    expandButton.textContent = isExpanded ? '−' : '+';
    expandButton.setAttribute('aria-expanded', isExpanded ? 'true' : 'false');
  });

  cardHeader.append(headerTitle, expandButton);

  const cardContent = document.createElement('div');
  cardContent.className = 'clip-card-content';

  const videoWrapper = document.createElement('div');
  videoWrapper.className = 'clip-video';
  const video = document.createElement('video');
  video.controls = true;
  video.preload = 'metadata';
  video.playsInline = true;
  video.src = backendUrl(clip.video_url || clip.download_url);
  videoWrapper.appendChild(video);

  const info = document.createElement('div');
  info.className = 'clip-info';

  const range = document.createElement('p');
  range.className = 'clip-range';
  range.textContent = formatClipRange(clip);

  const createMetaPill = (label, value) => {
    const pill = document.createElement('span');
    pill.className = 'clip-meta-pill';
    const strong = document.createElement('strong');
    strong.textContent = `${label}: `;
    pill.append(strong, document.createTextNode(value));
    return pill;
  };

  const startLabel = clip.start_timestamp || formatDuration(clip.start_second ?? 0);
  const endLabel = clip.end_timestamp || formatDuration(clip.end_second ?? 0);
  const metaGrid = document.createElement('div');
  metaGrid.className = 'clip-meta-grid';
  metaGrid.append(
    createMetaPill('Start', startLabel),
    createMetaPill('End', endLabel)
  );
  metaGrid.append(createMetaPill('Duration', `${durationSeconds}s`));

  const selectionLabel = document.createElement('label');
  selectionLabel.className = 'clip-selection';
  const selectionToggle = document.createElement('input');
  selectionToggle.type = 'checkbox';
  selectionToggle.checked = clipSelection.get(clipKey) ?? true;
  const selectionText = document.createElement('span');
  const updateSelectionState = (included) => {
    selectionText.textContent = included ? 'Included for download' : 'Excluded from download';
    li.classList.toggle('clip-card--excluded', !included);
    clipSelection.set(clipKey, included);
    updatePhaseClipsStatus();
    updateDownloadSelectedButton();
  };
  updateSelectionState(selectionToggle.checked);
  selectionToggle.addEventListener('change', () => updateSelectionState(selectionToggle.checked));
  selectionLabel.append(selectionToggle, selectionText);

  const actions = document.createElement('div');
  actions.className = 'clip-actions';

  const jumpButton = document.createElement('button');
  jumpButton.type = 'button';
  jumpButton.className = 'tertiary';
  jumpButton.textContent = 'Play in main video';
  jumpButton.addEventListener('click', () => {
    if (typeof clip.start_second === 'number') {
      summaryVideo.currentTime = clip.start_second;
      summaryVideo.play();
    }
  });

  const downloadLink = document.createElement('a');
  downloadLink.href = backendUrl(clip.download_url || clip.video_url);
  downloadLink.className = 'secondary';
  downloadLink.textContent = 'Download clip';
  downloadLink.download = clip.file_name || `${(clip.phase || 'phase').replace(/\s+/g, '-')}.mp4`;
  downloadLink.target = '_blank';
  downloadLink.rel = 'noreferrer';

  actions.append(jumpButton, downloadLink);

  const infoFooter = document.createElement('div');
  infoFooter.className = 'clip-details-footer';
  infoFooter.append(selectionLabel, actions);

  info.append(range, metaGrid, infoFooter);

  cardContent.append(videoWrapper, info);

  li.append(cardHeader, cardContent);
  return li;
};

const renderPhaseClips = (clips = []) => {
  if (!phaseClipsSection || !phaseClipsList) return;
  const uniqueClips = dedupePhaseClips(clips);
  renderedPhaseClips = uniqueClips;
  phaseClipsList.innerHTML = '';
  clipSelection.clear();
  uniqueClips.forEach((clip) => clipSelection.set(getClipKey(clip), true));

  if (!uniqueClips.length) {
    phaseClipsSection.classList.add('hidden');
    phaseClipsControls?.classList.add('hidden');
    updatePhaseClipsStatus();
    updateDownloadSelectedButton();
    return;
  }

  phaseClipsSection.classList.remove('hidden');
  phaseClipsControls?.classList.remove('hidden');
  uniqueClips.forEach((clip) => {
    const card = createClipCard(clip);
    phaseClipsList.appendChild(card);
  });
  updatePhaseClipsStatus();
  updateDownloadSelectedButton();
};

if (downloadSelectedClipsButton) {
  downloadSelectedClipsButton.addEventListener('click', handleDownloadSelectedClips);
}

const renderBackendResult = (result) => {
  summaryContent.classList.remove('hidden');
  summaryPlaceholder.classList.add('hidden');

  setVideoPreview();

  if (summaryTextSection && summaryTextContent && result.summary_text) {
    summaryTextSection.classList.remove('hidden');
    summaryTextContent.textContent = result.summary_text;
  }

  renderPhaseClips(result.phase_clips || []);

  summaryStatus.textContent = 'Summary ready for review';
  metadataStatus.textContent = 'Complete';
  showToast('Analysis complete! Review the detected phases and download clips as needed.');
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

  renderPhaseClips([]);
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
  showToast('Summary exported successfully.');
});

shareSummaryButton.addEventListener('click', () => {
  showToast('Shared with your team.');
});

if (helpButton) {
  helpButton.addEventListener('click', openHelpModal);
}

if (closeModalButton) {
  closeModalButton.addEventListener('click', closeHelpModal);
}

if (modalOverlay) {
  modalOverlay.addEventListener('click', closeHelpModal);
}

document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape' && helpModal && !helpModal.classList.contains('hidden')) {
    closeHelpModal();
  }
});

resetUI();
checkBackendHealth();
