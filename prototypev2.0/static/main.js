const hamburgerBtn = document.getElementById('hamburger-btn');
const mobileNav = document.getElementById('mobile-nav');

const video = document.getElementById('video-feed-dash');
const emotionDisplay    = document.getElementById('emotion-display');
const confidenceDisplay = document.getElementById('confidence-display');
const canvasElements = {
    dashboard   : document.getElementById('video-canvas-dash'),
    comparison  : document.getElementById('video-canvas-comp')
}
let canvas = null;
let ctx = null;

let videoStream = null;

/*========== NAVIGATION ========== */
function switchPage(link) {
    document.querySelectorAll('.nav-link').forEach(link => link.classList.remove('active'));
    document.querySelectorAll('.page-section').forEach(section => section.classList.remove('active'));

    document.getElementById(link).classList.add('active');
    document.getElementById(link + '-page').classList.add('active');

    // Sync mobile nav active state
    document.querySelectorAll('.mobile-nav-list .nav-link').forEach(el => {
        el.classList.toggle('active', el.id === `mob-${link}`)
    });

    // Stop inference and reset button when switching pages
    stopInference();

    document.querySelectorAll('.start-button').forEach(btn => {
        btn.textContent = link === 'comparison' ? 'Start Comparison' : 'Start Logging'
    })
    if (link === 'analysis') {
        requestAnimationFrame(() => renderChart());
        return
    }

    if (canvasElements[link]) {
        canvas = canvasElements[link];
        initCanvas();
        ctx = canvas.getContext('2d');  
    }
}

/*========== MOBILE NAV ========== */
function closeMobileNav() {
    mobileNav.classList.remove('open');
    hamburgerBtn.classList.remove('open');
    hamburgerBtn.setAttribute('aria-expanded', false);
}

hamburgerBtn.addEventListener('click', ()=> {
    const isOpen = mobileNav.classList.toggle('open');
    hamburgerBtn.classList.toggle('open', isOpen);
    hamburgerBtn.setAttribute('aria-expanded', isOpen);
});

document.addEventListener('click', (e)=> {
    if (!hamburgerBtn.contains(e.target) && !mobileNav.contains(e.target)) {
        closeMobileNav();
    }
})

/*========== CONFIG ========== */
const CONFIG = {
    inferenceInterval   : 250, //  between frames sent to server
    minConfidence       : .50,
    streamWidth         : 640,
    streamHeight        : 480,
    captureWidth        : 320,  //  downscaled before sending
    captureHeight       : 240,
    jpegQuality         : .6
};

/*========== CANVAS / CAMERA ========== */
function initCanvas() {
    if (!videoStream) return;
    // canvas dimensions should match stream dimensions
    canvas.width = videoStream.getVideoTracks()[0].getSettings().width || CONFIG.streamWidth;
    canvas.height = videoStream.getVideoTracks()[0].getSettings().height || CONFIG.streamHeight;
}

function drawLoop() {
    // Draw the raw onto the current active canvas when idle, or let the handler override it
    if (canvas && ctx && video.readyState >= 2) {
        drawFrame(
            video, ctx, canvas,
            lastResult?.bbox || null,
            lastResult ? (emotionColors[lastResult.label] || '#ffffff') : null,
            lastResult?.landmarks || null
        );
    }
    requestAnimationFrame(drawLoop);
}

async function initCamera() {
    try {
        // Requests the stream only once
        const stream = await navigator.mediaDevices.getUserMedia({ 
            video: {
                width       : { ideal: CONFIG.streamWidth },
                height      : { ideal: CONFIG.streamHeight },
                frameRate   : { ideal: 30, max: 30 }
            }
        });
        
        // Set stream to the video element for frame capture
        videoStream = stream;
        video.srcObject = stream;

        video.onloadedmetadata = () => {
            initCanvas();
            drawLoop();
        }

    } catch (err) {
        alert('Camera error: ' + err.message);
    }
}

/*========== INFERENCE ========== */
let isRunning       = false;
let inferenceTimer  = null;
let lastResult      = null;
let inferenceInFlight = false;  // guard: blocks overlapping requests
let placeholderRemoved = false;

// Tracks which page is active so sendFrame knows what mode to request
// 'dashboard'  = ensemble only     (compare: false)
// 'comparison' = ensemble + cnn    (compare: true)
let activePage = 'dashboard';

//  Off-screen canvas used only for downscaling before capture
const captureCanvas = document.createElement('canvas');
captureCanvas.width = CONFIG.captureWidth;
captureCanvas.height = CONFIG.captureHeight;
const captureCtx = captureCanvas.getContext('2d');

const logList = document.getElementById('log-list');
const ensembleLabel      = document.querySelector('#ensemble-log .emotion-label');
const ensembleConfidence = document.querySelector('#ensemble-log .emotion-confidence');
const cnnLabel      = document.querySelector('#cnn-log .emotion-label');
const cnnConfidence = document.querySelector('#cnn-log .emotion-confidence');

const emotionColors = {
    Happy       : '#39ffb4', 
    Sad         : '#ff55aa', 
    Fear        : '#aa55ff',
    Angry       : '#ff5555', 
    Disgust     : '#8DB600', 
    Surprise    : '#FF6B00', 
    Neutral     : '#55AAFF'
};

const EMOTION_ORDER = ['Happy', 'Neutral', 'Sad', 'Fear', 'Angry', 'Disgust', 'Surprise']

function renderEmotionBars(containerId, probs) {
    const container = document.getElementById(containerId);
    if (!container) return;

    if(!probs) {
        container.innerHTML = '<span class="emotion-bars-placeholder">No data</span>';
        return;
    }

    container.innerHTML = EMOTION_ORDER.map(emotion => {
        const val = probs[emotion] ?? 0;
        const pct = (val * 100).toFixed(1);
        const color = emotionColors[emotion] || '#aaa';
        return `
            <div class="emotion-bar-row">
                <span class="emotion-bar-label">${emotion}</span>
                <div class="emotion-bar-track">
                    <div class="emotion-bar-fill" style="width:${pct}%;background:${color}"></div>
                </div>
                <span class="emotion-bar-pct">${pct}%</span>
            </div>`;
    }).join('');
}



async function sendFrame() {
    if (inferenceInFlight) return;
    if (!isRunning || !videoStream || video.readyState < 2) return;

    inferenceInFlight = true;

    // Downscale frame to capture resolution before sending
    captureCtx.drawImage(video, 0, 0, CONFIG.captureWidth, CONFIG.captureHeight);
    const base64Frame = captureCanvas.toDataURL('image/jpeg', CONFIG.jpegQuality);

    const isCompare = activePage === 'comparison';
    const t0 = performance.now();   // Timestamp for measuring round-trip inference time

    try {
        const response = await fetch('/predict', {
            method  : 'POST',
            headers : { 'Content-Type': 'application/json'},
            body    : JSON.stringify({
                frame   : base64Frame,
                compare : isCompare
            }) 
        });

        const inferenceMs = (performance.now() - t0).toFixed(0);
        const result      = await response.json();

        handleResult(result, inferenceMs, isCompare);
    } catch (err) {
        console.error('Inference error:', err);
    } finally {
        inferenceInFlight = false;
    }
}

/*========== RESULT HANDLERS ========== */
function handleResult(result, inferenceMs, isCompare) {
    if (isCompare) {
        handleComparisonResult(result, inferenceMs);
    } else {
        handleDashboardResult(result, inferenceMs);
    }
}

// Dashboard: ensemble prediction only
function handleDashboardResult(result, inferenceMs) {
    if (result.label === 'No Face' || result.label === 'Error') {
        lastResult = null;
        emotionDisplay.textContent      = '--';
        confidenceDisplay.textContent   = 'No face detected';
        return;
    }

    lastResult = result;

    emotionDisplay.textContent = result.label;
    confidenceDisplay.textContent = 
        (result.confidence * 100).toFixed(1) + '%';

    renderEmotionBars('dash-bars-list', result.probs || null);
    
    // Append to session log every prediction
    appendSessionLog(result.label, result.confidence, inferenceMs);
    collectCandidate();
}

// Session log: one row per prediction, newest on top
const MAX_LOG_ENTRIES = 100; // cap so the list never grows unbounded

class Log {
    constructor(data, next = null) {
        this.data = data;
        this.next = next;
    }
}

class SessionLog {
    constructor() {
        this.head = null;
        this.size = 0;
    }

    prepend(data) {
        this.head = new Log(data, this.head);
        this.size++;
    }
}

const sessionLog = new SessionLog();

function appendSessionLog(label, confidence, inferenceMs) {
    if (!logList) return;

    // Remove the static placeholder on the very first real entry
    if (!placeholderRemoved) {
        const placeholder = logList.querySelector('[data-placeholder]');
        if (placeholder) placeholder.remove();
        placeholderRemoved = true;
    }

    // Timestamp e.g. "14:23:07"
    const time = new Date().toLocaleTimeString([], {
        hour    : '2-digit',
        minute  : '2-digit',
        second  : '2-digit'
    });

    const entry = document.createElement('div');
    entry.className = 'session-log-list-item';
    entry.style.borderLeftColor = emotionColors[label];
    entry.style.background = 'rgba(255, 255, 255, 0.03)'
    entry.innerHTML = `
        <span class="log-time">${time}</span>
        <span class="log-emotion" style="color:${emotionColors[label] || '#ffffff'}">${label}</span>
        <span class="log-confidence">${(confidence * 100).toFixed(1)}%</span>
        <span class="log-inference">${inferenceMs} ms</span>
    `;

    // Newest entry at the top
    logList.insertBefore(entry, logList.firstChild);
    sessionLog.prepend({ timestamp: time, label: label, confidence });

    // Trim oldest entries beyond the cap
    while (logList.children.length > MAX_LOG_ENTRIES) {
        logList.removeChild(logList.lastChild);
    }
}

// Comparison: ensemble vs CNN baseline, with inference time
function handleComparisonResult(result, inferenceMs) {
    const ensemble  = result.ensemble;
    const cnn       = result.cnn_only;

    // Keep bbox from ensemble result for canvas drawing
    if (ensemble && ensemble.label !== 'No Face' && ensemble.label !== 'Error') {
        lastResult = ensemble;
    } else {
        lastResult = null;
    }

    //  Ensemble panel (#ensemble-log)
    if (ensembleLabel)      ensembleLabel.textContent = ensemble?.label ?? '--';
    if (ensembleConfidence) ensembleConfidence.textContent = ensemble
        ? `${(ensemble.confidence * 100).toFixed(1)}% | ${inferenceMs} ms`
        : '0.0% | -- ms';
    
    renderEmotionBars('ensemble-bars-list', ensemble?.probs || null);

    // CNN baseline panel (#cnn-log)
    if (cnnLabel)       cnnLabel.textContent = cnn?.label ?? '--';
    if (cnnConfidence)  cnnConfidence.textContent = cnn
        ? `${(cnn.confidence * 100).toFixed(1)}% | ${inferenceMs} ms`
        : '0.0% | --ms';

    renderEmotionBars('cnn-bars-list', cnn?.probs || null);
}

/*========== DRAW ========== */
function drawFrame(video, context, canvas, bbox = null, colorHex = null, landmarks = null) {
    context.drawImage(video, 0, 0, canvas.width, canvas.height);

    const scaleX = canvas.width / CONFIG.captureWidth;
    const scaleY = canvas.height / CONFIG.captureHeight;

    if (bbox && bbox.length === 4) {
        const [x, y, w, h] = bbox;
        context.strokeStyle = colorHex;
        context.lineWidth = 4;
        context.strokeRect(x * scaleX, y * scaleY, w * scaleX, h * scaleY);
    }

    context.globalAlpha = 0.6

    if (landmarks && landmarks.length) {
        context.fillStyle = colorHex;
        for (const [x, y] of landmarks) {
            context.beginPath();
            context.arc(x * scaleX, y * scaleY, 1.5, 0, 2 * Math.PI);
            context.fill();
        }
    }

    context.globalAlpha = 1.0
}

/*========== START / STOP ========== */
function startInference() {
    if (isRunning) return;
    isRunning = true;
    inferenceTimer = setInterval(sendFrame, CONFIG.inferenceInterval);
    captureTimer   = setInterval(captureFrame, captureIntervalMs);

    setTimeout(() => captureFrame(), 3000);
}

function stopInference() {
    isRunning           = false;
    inferenceInFlight   = false;
    clearInterval(inferenceTimer);
    clearInterval(captureTimer);
    lastResult = null;
}

/*========== CHART ========== */
let emotionChartInstance = null;

function renderChart() {
    const ctxChart = document.getElementById('emotionChart').getContext('2d');
    const noDataMsg = document.getElementById('no-data-msg');

    if (sessionLog.size === 0) {
        if (emotionChartInstance) emotionChartInstance.destroy();
        if (noDataMsg) noDataMsg.style.display = 'block';
        return;
    }
    if (noDataMsg) noDataMsg.style.display = 'none';

    // 1. Aggregate Data
    const counts = {};
    let current = sessionLog.head;
    while (current) {
        const emo = current.data['label'];
        counts[emo] = (counts[emo] || 0) + 1;
        current = current.next;
    }

    const totalCount = sessionLog.size;
    const labels = Object.keys(counts);
    const rawData = Object.values(counts);

    // 2. Calculate percentages
    const percentages = rawData.map(count => ((count / totalCount) * 100).toFixed(1));
    const bgColors = labels.map(l => emotionColors[l] || '#cccccc');

    // 3. Destroy old chart if exists
    if (emotionChartInstance) {
        emotionChartInstance.destroy();
    };

    // 4. Create New Chart
    emotionChartInstance = new Chart(ctxChart, {
        type: 'doughnut',
        data: {
            labels: labels.map(l => l.toUpperCase()),
            datasets: [{
                data: rawData,
                backgroundColor: bgColors,
                borderWidth: 0,
                hoverOffset: 4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            layout: {
                padding: {
                    bottom: 8
                }
            },
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: { 
                        color: 'white',
                        padding: 16,
                        boxWidth: 12,
                        font: { size: 12 }
                    }
                },
                tooltip: {
                    callbacks: {
                        label: function (context) {
                            let label = context.label || '';
                            if (label) {
                                label += ': ';
                            }
                            const value = context.parsed;
                            const percentage = percentages[context.dataIndex];
                            return `${label} ${percentage}% (${value} total)`;
                        }
                    }
                }
            }
        }
    });
}

/*========== FRAME CAPTURES ========== */
let captureIntervalMs   = 5 * 60 * 1000;
let captureTimer        = null;
let captureBuffer       = [];           // rolling buffer of recent captures
let candidateBuffer     = [];           // rolling candidates collected during interval
const MAX_CAPTURES      = 7;
const MIN_CONFIDENCE    = 0.55;         // only candidates above this threshold

function updateCaptureInterval(minutes) {
    captureIntervalMs = parseInt(minutes) * 60 * 1000;
    if (isRunning) {
        clearInterval(captureTimer);
        captureTimer = setInterval(captureFrame, captureIntervalMs);
    }
}

// Called every inference frame — collects candidates silently
function collectCandidate() {
    if (!isRunning || !lastResult || !videoStream) return;
    if (video.readyState < 2) return;
    if (lastResult.label === 'No Face' || lastResult.label === 'Error') return;
    if (lastResult.confidence < MIN_CONFIDENCE) return;

    // Draw current frame
    const snap    = document.createElement('canvas');
    snap.width    = CONFIG.captureWidth;
    snap.height   = CONFIG.captureHeight;
    const snapCtx = snap.getContext('2d');
    snapCtx.drawImage(video, 0, 0, snap.width, snap.height);

    // Crop face bbox with padding
    let dataUrl;
    if (lastResult.bbox && lastResult.bbox.length === 4) {
        const [bx, by, bw, bh] = lastResult.bbox;

        const padX = Math.round(0.5 * bw);  // horizontal padding
        const padY = Math.round(0.6 * bh); // vertical padding

        const cx  = Math.max(0, bx - padX);
        const cy  = Math.max(0, by - padY);
        const cw  = Math.min(bw + 2 * padX, snap.width  - cx);
        const ch  = Math.min(bh + 2 * padY, snap.height - cy);

        const crop    = document.createElement('canvas');
        crop.width    = cw;
        crop.height   = ch;
        crop.getContext('2d').drawImage(snap, cx, cy, cw, ch, 0, 0, cw, ch);
        dataUrl = crop.toDataURL('image/jpeg', 0.8);
    } else {
        dataUrl = snap.toDataURL('image/jpeg', 0.8);
    }

    candidateBuffer.push({
        dataUrl,
        label      : lastResult.label,
        confidence : lastResult.confidence,
        time       : new Date().toLocaleTimeString([], {
            hour: '2-digit', minute: '2-digit', second: '2-digit'
        }),
        color: emotionColors[lastResult.label] || '#ffffff'
    });
}

// Called every interval — picks top 5 unique from candidates
function captureFrame() {
    if (candidateBuffer.length === 0) return;

    // Pick best (highest confidence) per unique emotion label
    const bestPerEmotion = {};
    candidateBuffer.forEach(entry => {
        if (!bestPerEmotion[entry.label] ||
            entry.confidence > bestPerEmotion[entry.label].confidence) {
            bestPerEmotion[entry.label] = entry;
        }
    });

    // Sort by confidence descending, take top MAX_CAPTURES
    const top5 = Object.values(bestPerEmotion)
        .sort((a, b) => b.confidence - a.confidence)
        .slice(0, MAX_CAPTURES);

    captureBuffer = top5;
    candidateBuffer = [];   // reset for next interval
    renderCaptures();
}

function renderCaptures() {
    const grid  = document.getElementById('captures-grid');
    const empty = document.getElementById('captures-empty');
    if (!grid) return;

    if (captureBuffer.length === 0) {
        if (empty) empty.style.display = 'flex';
        return;
    }

    if (empty) empty.style.display = 'none';

    // Clear existing cards (keep empty placeholder in DOM)
    grid.querySelectorAll('.capture-card').forEach(c => c.remove());

    captureBuffer.forEach(entry => {
        const card = document.createElement('div');
        card.className = 'capture-card';
        card.style.borderColor = entry.color + '55'; // subtle tint

        card.innerHTML = `
            <img src="${entry.dataUrl}" alt="${entry.label}" />
            <span class="capture-card__label" style="color:${entry.color}">
                ${entry.label}
            </span>
            <span class="capture-card__confidence">
                ${(entry.confidence * 100).toFixed(1)}%
            </span>
            <span class="capture-card__time">${entry.time}</span>
        `;
        grid.appendChild(card);
    });
}

/*========== EXPORT ========== */
function exportData(type) {
    if (sessionLog.size === 0) {
        alert("No data collected yet! Go to Dashboard and start logging.");
        return;
    }

    if (type === 'csv') {
        // Raw CSV
        let content = "Timestamp,Emotion,Confidence\n";
        let current = sessionLog.head;
        while (current) {
            content += `${current.data.timestamp},${current.data.label},${(current.data.confidence * 100).toFixed(2)}%\n`;
            current = current.next;
        }
        downloadFile(content, 'emotion_log.csv');

    } else if (type === 'psych') {
        // Clinical log — structured, meaningful
        const now       = new Date();
        const dateStr   = now.toLocaleDateString([], { year: 'numeric', month: 'long', day: 'numeric' });
        const timeStr   = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

        // Aggregate stats
        const counts    = {};
        const confByEmo = {};
        let current     = sessionLog.head;
        let totalConf   = 0;

        while (current) {
            const { label, confidence } = current.data;
            counts[label]    = (counts[label] || 0) + 1;
            confByEmo[label] = (confByEmo[label] || []);
            confByEmo[label].push(confidence);
            totalConf += confidence;
            current = current.next;
        }

        const total       = sessionLog.size;
        const avgConf     = (totalConf / total * 100).toFixed(1);
        const dominant    = Object.entries(counts).sort((a, b) => b[1] - a[1])[0];
        const dominantPct = (dominant[1] / total * 100).toFixed(1);

        // Build report
        let content = '';
        content += `CLINICAL FACIAL EXPRESSION ANALYSIS REPORT\n`;
        content += `${'='.repeat(50)}\n\n`;
        content += `Date:              ${dateStr}\n`;
        content += `Time:              ${timeStr}\n`;
        content += `Total Observations: ${total}\n`;
        content += `Avg Confidence:    ${avgConf}%\n\n`;

        content += `SESSION SUMMARY\n`;
        content += `${'-'.repeat(50)}\n`;
        content += `Dominant Emotion:  ${dominant[0]} (${dominantPct}% of session)\n\n`;

        content += `EMOTION BREAKDOWN\n`;
        content += `${'-'.repeat(50)}\n`;
        content += `Emotion,Count,Percentage,Avg Confidence\n`;

        Object.entries(counts)
            .sort((a, b) => b[1] - a[1])
            .forEach(([emotion, count]) => {
                const pct     = (count / total * 100).toFixed(1);
                const avgC    = (confByEmo[emotion].reduce((a, b) => a + b, 0) / confByEmo[emotion].length * 100).toFixed(1);
                content += `${emotion},${count},${pct}%,${avgC}%\n`;
            });

        content += `\nDETAILED OBSERVATION LOG\n`;
        content += `${'-'.repeat(50)}\n`;
        content += `Timestamp,Emotion,Confidence\n`;

        current = sessionLog.head;
        while (current) {
            content += `${current.data.timestamp},${current.data.label},${(current.data.confidence * 100).toFixed(2)}%\n`;
            current = current.next;
        }

        content += `\n${'='.repeat(50)}\n`;
        content += `Generated by Ensemble CNN-CatBoost FER System\n`;
        content += `For professional use only\n`;

        downloadFile(content, 'clinical_data_log.txt');
    }
}

// Helper to avoid repeating blob/download logic
function downloadFile(content, filename) {
    const blob = new Blob(["\ufeff" + content], { type: 'text/plain;charset=utf-8;' });
    const url  = window.URL.createObjectURL(blob);
    const a    = document.createElement('a');
    a.href     = url;
    a.download = filename;
    a.click();
    window.URL.revokeObjectURL(url);
    alert(`Exported successfully as ${filename}!`);
}

/*========== INIT ========== */
window.addEventListener('load', () => {
    activePage = 'dashboard';
    canvas     = canvasElements['dashboard'];
    ctx        = canvas.getContext('2d');
    initCamera();

    document.querySelectorAll('.start-button').forEach(btn => {
        btn.addEventListener('click', () => {
            if (!isRunning) {
                // Detect which page this button lives on
                activePage = btn.closest('#comparison-page') ? 'comparison' : 'dashboard';

                // Sync canvas context to the active page
                canvas = canvasElements[activePage] || canvasElements['dashboard'];
                ctx = canvas.getContext('2d');
                initCanvas();

                startInference();
                btn.textContent = activePage === 'comparison' ? 'Stop Comparison' : 'Stop';
            } else {
                stopInference();
                btn.textContent = activePage === 'comparison' ? 'Start Comparison' : 'Start Logging';
            }
        })
    });
});