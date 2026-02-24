const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const uploadCard = document.getElementById('upload-card');
const fileInfo = document.getElementById('file-info');
const filenameSpan = document.getElementById('filename');
const processBtn = document.getElementById('process-btn');
const clearBtn = document.getElementById('clear-btn');
const dashboard = document.getElementById('dashboard');
const results = document.getElementById('results');

const progressBar = document.getElementById('progress-bar');
const percentText = document.getElementById('percent-text');
const frameCount = document.getElementById('frame-count');
const statusText = document.getElementById('status-text');
const logs = document.getElementById('logs');

let currentFile = null;

// File Selection
dropZone.onclick = () => fileInput.click();

fileInput.onchange = (e) => {
    if (e.target.files.length > 0) {
        handleFileSelect(e.target.files[0]);
    }
};

dropZone.ondragover = (e) => {
    e.preventDefault();
    dropZone.classList.add('dragover');
};

dropZone.ondragleave = () => {
    dropZone.classList.remove('dragover');
};

dropZone.ondrop = (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    if (e.dataTransfer.files.length > 0) {
        handleFileSelect(e.dataTransfer.files[0]);
    }
};

function handleFileSelect(file) {
    currentFile = file;
    filenameSpan.innerText = file.name;
    fileInfo.classList.remove('hidden');
    dropZone.classList.add('hidden');
}

clearBtn.onclick = () => {
    currentFile = null;
    fileInfo.classList.add('hidden');
    dropZone.classList.remove('hidden');
    fileInput.value = '';
};

// Processing logic
processBtn.onclick = async () => {
    if (!currentFile) return;

    try {
        addLog(`Uploading ${currentFile.name}...`);
        const formData = new FormData();
        formData.append('file', currentFile);

        const uploadRes = await fetch('/upload', {
            method: 'POST',
            body: formData
        });

        if (!uploadRes.ok) throw new Error('Upload failed');

        const { filename } = await uploadRes.json();
        addLog('Upload complete. Starting pipeline...');

        dashboard.classList.remove('hidden');
        uploadCard.classList.add('hidden');

        // Start processing
        const processRes = await fetch(`/process/${filename}`, { method: 'POST' });
        if (!processRes.ok) throw new Error('Processing failed to start');

        connectWebSocket(filename);

    } catch (err) {
        addLog(`Error: ${err.message}`);
        statusText.innerText = 'Error';
    }
};

function connectWebSocket(videoId) {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const ws = new WebSocket(`${protocol}//${window.location.host}/ws/progress/${videoId}`);

    // Set live feed source
    const liveFeed = document.getElementById('live-feed');
    liveFeed.src = `/stream/${videoId}`;
    liveFeed.classList.remove('hidden');

    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        updateUI(data);
    };

    ws.onerror = (err) => {
        addLog('WebSocket Connection Error');
    };

    ws.onclose = () => {
        addLog('Processing finished.');
    };
}

function updateUI(data) {
    if (data.status) statusText.innerText = data.status;
    if (data.percentage !== undefined) {
        progressBar.style.width = `${data.percentage}%`;
        percentText.innerText = `${data.percentage}%`;
    }
    if (data.frame !== undefined) {
        frameCount.innerText = `${data.frame} / ${data.total || '?'}`;
    }

    if (data.status === 'Complete') {
        showResults(data.video_id);
    }
}

function showResults(videoId) {
    const outputPrefix = videoId.split('.')[0] + '_vrg';
    const videoUrl = `/outputs/${outputPrefix}.mp4`;
    const jsonUrl = `/outputs/${outputPrefix}.json`;

    document.getElementById('input-video').src = `/static/uploads/${videoId}`;
    document.getElementById('output-video').src = videoUrl;
    document.getElementById('download-video').href = videoUrl;
    document.getElementById('download-json').href = jsonUrl;

    results.classList.remove('hidden');
    addLog('Results ready.');

    // Jump to results
    results.scrollIntoView({ behavior: 'smooth' });
}

function addLog(msg) {
    const entry = document.createElement('div');
    entry.className = 'log-entry';
    entry.innerText = `[${new Date().toLocaleTimeString()}] ${msg}`;
    logs.appendChild(entry);
    logs.scrollTop = logs.scrollHeight;
}
