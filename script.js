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
const highlightsList = document.querySelector('#highlights-list');
const toast = document.querySelector('#toast');
const downloadSummaryButton = document.querySelector('#download-summary');
const shareSummaryButton = document.querySelector('#share-summary');
const localModeIndicator = document.querySelector('#local-mode-indicator');

let uploadedFile = null;
let mockProcessingTimeout = null;

const isDesktopShell = Boolean(window?.desktopApp?.isElectron);

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

const resetUI = () => {
  uploadedFile = null;
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
  highlightsList.innerHTML = '';
  if (mockProcessingTimeout) {
    clearTimeout(mockProcessingTimeout);
    mockProcessingTimeout = null;
  }
};

const describeEnvironment = () => {
  if (!localModeIndicator) return;
  localModeIndicator.textContent = isDesktopShell
    ? 'Running in secure desktop shell'
    : 'Running locally in your browser';
};

const showToast = (message, duration = 4000) => {
  toast.textContent = message;
  toast.classList.add('visible');
  setTimeout(() => toast.classList.remove('visible'), duration);
};

const simulateSummary = () => {
  summaryContent.classList.remove('hidden');
  summaryPlaceholder.classList.add('hidden');
  summaryStatus.textContent = 'Summary ready for review';

  summaryVideo.src = 'https://storage.googleapis.com/coverr-main/mp4/Mt_Baker.mp4';
  summaryVideo.load();

  highlightsList.innerHTML = '';
  const highlights = [
    { timestamp: '00:02:14', description: 'Initial incision and access established' },
    { timestamp: '00:07:52', description: 'Critical vessel identified and secured' },
    { timestamp: '00:12:08', description: 'Anastomosis inspection and confirmation' }
  ];

  for (const { timestamp, description } of highlights) {
    const li = document.createElement('li');
    li.innerHTML = `
      <span class="timestamp">${timestamp}</span>
      <span class="description">${description}</span>
      <button type="button" class="tertiary" data-jump="${timestamp}">Review</button>
    `;
    highlightsList.appendChild(li);
  }

  highlightsList.querySelectorAll('button[data-jump]').forEach((button) => {
    button.addEventListener('click', () => {
      const [hours, minutes, seconds] = button.dataset.jump.split(':').map(Number);
      const timeInSeconds = hours * 3600 + minutes * 60 + seconds;
      summaryVideo.currentTime = timeInSeconds;
      summaryVideo.play();
    });
  });

  showToast('Summary complete. Review key moments and export when ready.');
};

const simulateProcessing = () => {
  summaryStatus.textContent = 'Analyzing procedure footage…';
  metadataStatus.textContent = 'Processing';
  showToast('Upload received. Starting AI-assisted summarization.');

  if (mockProcessingTimeout) {
    clearTimeout(mockProcessingTimeout);
  }

  mockProcessingTimeout = setTimeout(simulateSummary, 3000);
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
    const duration = tempVideo.duration;
    const minutes = Math.floor(duration / 60);
    const seconds = Math.round(duration % 60).toString().padStart(2, '0');
    const hours = Math.floor(minutes / 60);
    const displayMinutes = (minutes % 60).toString().padStart(2, '0');
    const displayHours = hours ? `${hours}:` : '';
    metadataDuration.textContent = `${displayHours}${displayMinutes}:${seconds}`;
    URL.revokeObjectURL(fileURL);
  });
});

startProcessingButton.addEventListener('click', () => {
  if (!uploadedFile) return;
  startProcessingButton.disabled = true;
  clearSelectionButton.disabled = true;
  simulateProcessing();
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
describeEnvironment();
