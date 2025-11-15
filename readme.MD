# Surgical Video Summarizer Frontend

This project provides a desktop-style web interface for uploading surgical procedure recordings, simulating AI processing, and previewing an auto-generated highlight reel. The UI is implemented with vanilla HTML, CSS, and JavaScript, so no build tooling is required.

## Prerequisites

- Any modern web browser (Chrome, Firefox, Safari, Edge)
- A local static file server (optional but recommended). Python is preinstalled on most systems and includes a simple HTTP server.

## Getting Started

Choose the launch option that best matches your workflow and compliance requirements. Every mode keeps footage on the local workstation—no uploads are performed.

### Option A: Desktop shell (recommended for long recordings)
1. Install [Node.js](https://nodejs.org/) 18 or newer.
2. Open a terminal in the project root and run:

   ```bash
   cd desktop
   npm install
   npm start
   ```

   This boots an Electron window that loads `index.html` directly from disk while providing a native desktop frame.

### Option B: Serve with a local HTTP server
Serving the files removes browser restrictions around local file access and ensures media playback works as expected.

1. Clone or download this repository.
2. Open a terminal in the project directory.
3. Start a static server. Examples:
   - **Python 3:** `python -m http.server 8000`
   - **Node.js (http-server):** `npx http-server -p 8000`
4. Visit `http://localhost:8000` in your browser.

### Option C: Open the file directly
1. Clone or download this repository.
2. Double-click `index.html` (or open it via your browser's **File → Open File…** menu).
3. Some browsers limit video playback from the `file://` scheme. If you run into restrictions, switch to Option A or B.


## Starting backend
In the root directory "uvicorn backend.main:app --reload"

## Usage

1. Click **Upload Surgical Video** to select a video file from your computer.
2. Review the extracted metadata and enter optional procedure notes.
3. Press **Generate Summary** to simulate the AI workflow. The interface will display progress, highlight cards, and a condensed playback preview.
4. Use the export controls to simulate downloading the highlight reel or copying timestamps for documentation.

## Frontend ↔ Backend Flow
- `POST /upload` saves the selected video locally and returns a `file_id` the UI keeps in memory.
- `POST /process` starts a background job for that `file_id` and returns a `job_id`.
- `GET /jobs/{job_id}` reports job status (`queued`, `processing`, `completed`, `failed`) plus a progress percent; the frontend polls this endpoint until completion.
- `GET /jobs/{job_id}/result` delivers the final summary text, highlight metadata, and sampled frame previews once the job is done.

**Example:** After uploading `laparoscopy.mp4`, the frontend receives `file_id=file_ab12cd34`. It immediately calls `/process` and gets `job_id=job_ef56gh78`. While the backend slices frames and builds the stub summary, the UI polls `/jobs/job_ef56gh78`. When the status flips to `completed`, the UI requests `/jobs/job_ef56gh78/result` and renders the returned summary and highlight cards.

## Project Structure

- `index.html` – Layout markup and component structure for the desktop UI.
- `styles.css` – Visual design, theming, and layout rules.
- `script.js` – Client-side logic for handling uploads, simulating processing, updating the UI, and managing playback interactions.
- `desktop/` – Minimal Electron wrapper (`main.js`, `preload.js`, `package.json`) for packaging the UI as an installable desktop app.

## Troubleshooting

- If you run into playback restrictions when opening `index.html` directly, switch to the Electron shell (Option A) or a local HTTP server (Option B).
- Large video files may take longer to scan for metadata; try smaller samples during UI smoke tests.
- Clear your browser cache if you make local changes to the frontend files and do not see them reflected immediately.

## Why offer a desktop option?

- **Eliminate upload bottlenecks:** Multi-gigabyte source files never leave the device, so there's no waiting on network transfers.
- **Privacy & compliance:** Patient-identifiable footage stays on-premises, simplifying HIPAA/GDPR considerations.
- **Native ergonomics:** The Electron shell can be extended with OS-level integrations, accelerator keys, or PACS tooling without rewriting the core UI.

## License

This project is provided as-is for demonstration purposes. Adapt it freely to match your production environment and compliance requirements.
