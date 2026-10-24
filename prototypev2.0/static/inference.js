// ─── INFERENCE STATE ──────────────────────────────────────
let isRunning         = false;
let inferenceTimer    = null;
let lastResult        = null;
let inferenceInFlight = false;
let activePage        = 'dashboard';
let sessionStartTime  = null;

// Dashboard rolling counters
let dashCount   = 0;
let dashConfSum = 0;

// Comparison counters
let sessionFrames    = 0;
let ensLowConf       = 0;
let cnnLowConf       = 0;
let ensEmotionCounts = {};
let cnnEmotionCounts = {};
let ensConfSum       = 0;
let cnnConfSum       = 0;

const captureCanvas = document.createElement('canvas');
captureCanvas.width  = CONFIG.captureWidth;
captureCanvas.height = CONFIG.captureHeight;
const captureCtx = captureCanvas.getContext('2d');

// ─── RESET ────────────────────────────────────────────────
function resetSessionCounters() {
    sessionFrames = 0; ensLowConf = 0; cnnLowConf = 0;
    ensEmotionCounts = {}; cnnEmotionCounts = {};
    ensConfSum = 0; cnnConfSum = 0;
    dashCount = 0; dashConfSum = 0;
    placeholderRemoved = false;
    clearMetricHistory();
    sessionStartTime = Date.now(); 
}

// ─── SEND FRAME ───────────────────────────────────────────
async function sendFrame() {
    if (inferenceInFlight || !isRunning || !videoStream) return;
    if (video.readyState < 2) return;

    inferenceInFlight = true;
    captureCtx.drawImage(video, 0, 0, CONFIG.captureWidth, CONFIG.captureHeight);
    const base64Frame = captureCanvas.toDataURL('image/jpeg', CONFIG.jpegQuality);
    const isCompare   = activePage === 'comparison';
    const t0          = performance.now();

    try {
        const response = await fetch('/predict', {
            method : 'POST',
            headers: { 'Content-Type': 'application/json' },
            body   : JSON.stringify({ frame: base64Frame, compare: isCompare })
        });
        const inferenceMs = (performance.now() - t0).toFixed(0);
        const result      = await response.json();

        isCompare
            ? handleComparisonResult(result, inferenceMs)
            : handleDashboardResult(result, inferenceMs);

        if (CONFIG.adaptiveInterval && inferenceMs > 1000) {
            clearInterval(inferenceTimer);
            inferenceTimer = setInterval(sendFrame, Math.min(inferenceMs * 0.8, 3000));
        }
    } catch (err) {
        console.error('Inference error:', err);
    } finally {
        inferenceInFlight = false;
    }
}

// ─── DASHBOARD HANDLER ────────────────────────────────────
function handleDashboardResult(result, inferenceMs) {
    const emotionDisplay    = document.getElementById('emotion-display');
    const confidenceDisplay = document.getElementById('confidence-display');

    if (result.label === 'No Face' || result.label === 'Error') {
        lastResult = null;
        emotionDisplay.textContent    = '--';
        confidenceDisplay.textContent = 'No face detected';
        updateDistanceBadge('dashboard', null);
        return;
    }

    lastResult = result;
    emotionDisplay.textContent    = result.label;
    confidenceDisplay.textContent = (result.confidence * 100).toFixed(1) + '%';

    renderEmotionBars('dash-bars-list', result.probs || null);

    const metrics = computeFacialMetrics(result.landmarks);
    appendFacialMetrics(metrics);
    appendDistanceMetric(result.distance_cm ?? null);
    updateDistanceBadge('dashboard', result.distance_cm ?? null);
    appendSessionLog(result.label, result.confidence, inferenceMs);
    collectCandidate();

    // Arousal tracking
    arousalHistory.push(AROUSAL_MAP[result.label] ?? 0.5);
    if (arousalHistory.length > MAX_HISTORY) arousalHistory.shift();

    if (metrics) {
        setTextContent('eye-contact-time', `${Math.round(metrics.eyeContact)}%`);
        setTextContent('smile-count',      `${Math.round(metrics.smileScore)}`);
        setTextContent('microexp-count',   metrics.micro.reduce((s, v) => s + v, 0).toFixed(0));
    }

    dashCount++;
    dashConfSum += result.confidence;
    setTextContent('dashboard-avg-confidence', `${(dashConfSum / dashCount * 100).toFixed(1)}%`);
    setTextContent('dashboard-latency',        `${inferenceMs} ms`);
    setTextContent('dashboard-capture-count',  `${dashCount}`);
    setTextContent('dashboard-top-emotion',    result.label);

    if (activePage === 'analysis') requestAnimationFrame(renderChart);
}

// ─── COMPARISON HANDLER ───────────────────────────────────
function handleComparisonResult(result, inferenceMs) {
    const ensemble = result.ensemble;
    const cnn      = result.cnn_only;

    lastResult = (ensemble?.label !== 'No Face' && ensemble?.label !== 'Error') ? ensemble : null;

    updateDistanceBadge('comparison', result.distance_cm ?? null);

    setTextContent('comp-ens-label',      ensemble?.label ?? '--');
    setTextContent('comp-cnn-label',      cnn?.label      ?? '--');
    setTextContent('comp-ens-confidence', ensemble
        ? `${(ensemble.confidence * 100).toFixed(1)}% | ${inferenceMs} ms` : '0.0% | -- ms');
    setTextContent('comp-cnn-confidence', cnn
        ? `${(cnn.confidence * 100).toFixed(1)}% | ${inferenceMs} ms`      : '0.0% | -- ms');

    renderEmotionBars('ensemble-bars-list', ensemble?.probs || null);
    renderEmotionBars('cnn-bars-list',      cnn?.probs      || null);

    if (ensemble?.confidence !== undefined && cnn?.confidence !== undefined) {
        sessionFrames++;
        ensConfSum += ensemble.confidence;
        cnnConfSum += cnn.confidence;
        if (ensemble.confidence < 0.5) ensLowConf++;
        if (cnn.confidence      < 0.5) cnnLowConf++;

        if (ensemble.label && !['No Face','Error'].includes(ensemble.label))
            ensEmotionCounts[ensemble.label] = (ensEmotionCounts[ensemble.label] || 0) + 1;
        if (cnn.label && !['No Face','Error'].includes(cnn.label))
            cnnEmotionCounts[cnn.label] = (cnnEmotionCounts[cnn.label] || 0) + 1;
    }

    const safePct = v => v !== undefined ? `${(v * 100).toFixed(1)}%` : '--';
    const getDom  = counts => {
        if (!Object.keys(counts).length) return '--';
        const [label, count] = Object.entries(counts).sort((a, b) => b[1] - a[1])[0];
        return `${label} (${(count / sessionFrames * 100).toFixed(1)}%)`;
    };

    const updatePanel = (ids, confVal, confSum, lowVal, counts) => {
        setTextContent(ids.conf, safePct(confVal));
        if (sessionFrames > 0) setTextContent(ids.avg, safePct(confSum / sessionFrames));
        setTextContent(ids.low, lowVal);
        setTextContent(ids.dom, getDom(counts));
    };

    updatePanel(
        { conf:'comp-cnn-conf', avg:'comp-cnn-avg-conf', low:'comp-cnn-low', dom:'comp-cnn-dominant' },
        cnn?.confidence, cnnConfSum, cnnLowConf, cnnEmotionCounts
    );
    updatePanel(
        { conf:'comp-ensemble-conf', avg:'comp-ens-avg-conf', low:'comp-ens-low', dom:'comp-ens-dominant' },
        ensemble?.confidence, ensConfSum, ensLowConf, ensEmotionCounts
    );
}

// ─── EMOTION BARS ─────────────────────────────────────────
function renderEmotionBars(containerId, probs) {
    const container = document.getElementById(containerId);
    if (!container) return;

    if (!probs) {
        if (!container.querySelector('.emotion-bar-skeleton-list'))
            container.innerHTML = '<div class="emotion-bars-placeholder">No data</div>';
        return;
    }

    container.querySelector('.emotion-bar-skeleton-list')?.remove();

    container.innerHTML = EMOTION_ORDER.map(emotion => {
        const pct   = ((probs[emotion] ?? 0) * 100).toFixed(1);
        const color = emotionColors[emotion] || '#aaa';
        return `<div class="emotion-bar-row">
            <span class="emotion-bar-label">${emotion}</span>
            <div class="emotion-bar-track">
                <div class="emotion-bar-fill" style="width:${pct}%;background:${color}"></div>
            </div>
            <span class="emotion-bar-pct">${pct}%</span>
        </div>`;
    }).join('');
}

// ─── DISTANCE BADGE ───────────────────────────────────────
function updateDistanceBadge(page, cm) {
    // page = 'dashboard' → #distance-badge
    // page = 'comparison' → #distance-badge-comp
    const id    = page === 'comparison' ? 'distance-badge-comp' : 'distance-badge';
    const badge = document.getElementById(id);
    if (!badge) return;

    if (cm == null || cm <= 0) {
        badge.textContent       = 'Optimal Distance - cm';
        badge.style.background  = 'rgba(0,0,0,0.55)';
        badge.style.color       = '#fff';
        badge.style.borderColor = 'rgba(255,255,255,0.2)';
        return;
    }

    const zone = getDistanceZone(cm);
    badge.textContent       = `${cm} cm  —  ${zone.label}`;
    badge.style.background  = 'rgba(0,0,0,0.65)';
    badge.style.color       = zone.color;
    badge.style.borderColor = zone.color + '88';
}

// ─── START / STOP ─────────────────────────────────────────
function startInference() {
    if (isRunning) return;
    isRunning = true;
    sessionStartTime = Date.now();
    resetSessionCounters();

    ['cnn-bars-list', 'ensemble-bars-list'].forEach(id => {
        const el = document.getElementById(id);
        if (el && !el.querySelector('.emotion-bar-skeleton-list'))
            el.innerHTML = buildSkeletonHTML();
    });

    inferenceTimer = setInterval(sendFrame, CONFIG.inferenceInterval);
    captureTimer   = setInterval(captureFrame, captureIntervalMs);
    setTimeout(captureFrame, 3000);
}

function stopInference() {
    isRunning         = false;
    inferenceInFlight = false;
    clearInterval(inferenceTimer);
    clearInterval(captureTimer);
    lastResult = null;
    updateDistanceBadge('dashboard',   null);
    updateDistanceBadge('comparison',  null);
}