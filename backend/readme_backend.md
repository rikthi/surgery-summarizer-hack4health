	Backend 
    1.	Upload Video – Frontend sends a video to /upload; backend validates and uploads it to ImageKit.
	2.	Store Metadata – File info (URL, name, size) is saved in memory with a unique file_id.
	3.	Start Processing – /process creates a background job linked to that file_id and returns a job_id.
	4.	Simulate AI Task – A background task runs asynchronously, updates progress, and generates a fake summary (to be replaced afterwards).
	5.	Save Result – Processed output (summary, timeline, confidence) is stored in memory.
	6.	Check Status – Frontend polls /jobs/{job_id} for updates.
	7.	Get Summary – Once completed, /jobs/{job_id}/result returns the final summary and key steps.


Frameworks and Libraries Used
	•	FastAPI – Core web framework for building APIs
	•	Python-Dotenv – Loads environment variables from .env files
	•	ImageKit SDK – Handles video upload and storage to ImageKit CDN
	•	Pydantic – Validates request and response data models
	•	Asyncio – Enables asynchronous background task simulation
	•	BackgroundTasks (FastAPI) – Runs non-blocking AI job simulations
	•	JSONResponse – Formats structured JSON API responses

    Note 
    Backend uses a .env file to store sensitive credentials such as the ImageKit API keys. 
    How to use 
    - copy the example file and rename it to .env
    - open it and add your imageKit keys 
    - save the file 