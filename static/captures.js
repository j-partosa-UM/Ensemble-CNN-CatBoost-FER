// ─── FRAME CAPTURES ───────────────────────────────────────
let captureIntervalMs = 5 * 60 * 1000;
let captureTimer      = null;
let captureBuffer     = [];
let candidateBuffer   = [];
const MAX_CAPTURES    = 7;
const MIN_CONFIDENCE  = 0.55;

function updateCaptureInterval(minutes) {
    captureIntervalMs = parseInt(minutes) * 60 * 1000;
    if (isRunning) {
        clearInterval(captureTimer);
        captureTimer = setInterval(captureFrame, captureIntervalMs);
    }
}

function collectCandidate() {
    if (!isRunning || !lastResult || !videoStream) return;
    if (video.readyState < 2) return;
    if (['No Face','Error'].includes(lastResult.label)) return;
    if (lastResult.confidence < MIN_CONFIDENCE) return;

    const snap = document.createElement('canvas');
    snap.width  = CONFIG.captureWidth;
    snap.height = CONFIG.captureHeight;
    snap.getContext('2d').drawImage(video, 0, 0, snap.width, snap.height);

    let dataUrl;
    if (lastResult.bbox?.length === 4) {
        const [bx, by, bw, bh] = lastResult.bbox;
        const padX = Math.round(0.5 * bw);
        const padY = Math.round(0.6 * bh);
        const cx   = Math.max(0, bx - padX);
        const cy   = Math.max(0, by - padY);
        const cw   = Math.min(bw + 2*padX, snap.width  - cx);
        const ch   = Math.min(bh + 2*padY, snap.height - cy);
        const crop = document.createElement('canvas');
        crop.width  = cw; crop.height = ch;
        crop.getContext('2d').drawImage(snap, cx, cy, cw, ch, 0, 0, cw, ch);
        dataUrl = crop.toDataURL('image/jpeg', 0.8);
    } else {
        dataUrl = snap.toDataURL('image/jpeg', 0.8);
    }

    candidateBuffer.push({
        dataUrl,
        label     : lastResult.label,
        confidence: lastResult.confidence,
        time      : new Date().toLocaleTimeString([], { hour:'2-digit', minute:'2-digit', second:'2-digit' }),
        color     : emotionColors[lastResult.label] || '#fff'
    });
}

function captureFrame() {
    if (!candidateBuffer.length) return;

    const best = {};
    candidateBuffer.forEach(e => {
        if (!best[e.label] || e.confidence > best[e.label].confidence) best[e.label] = e;
    });

    captureBuffer   = Object.values(best).sort((a,b) => b.confidence-a.confidence).slice(0, MAX_CAPTURES);
    candidateBuffer = [];
    renderCaptures();
}

function renderCaptures() {
    const grid     = document.getElementById('captures-grid');
    const empty    = document.getElementById('captures-empty');
    const skeletons = document.getElementById('capture-placeholders');
    if (!grid) return;

    grid.querySelectorAll('.capture-card, .capture-skeleton').forEach(el => el.remove());

    if (!captureBuffer.length) {
        if (empty)    empty.style.display    = 'flex';
        if (skeletons) { skeletons.style.display = 'contents'; grid.appendChild(skeletons); }
        return;
    }

    if (empty)    empty.style.display    = 'none';
    if (skeletons) skeletons.style.display = 'none';

    captureBuffer.forEach(entry => {
        const card = document.createElement('div');
        card.className = 'capture-card';
        card.style.borderColor = entry.color + '55';
        card.innerHTML = `
            <img src="${entry.dataUrl}" alt="${entry.label}" />
            <span class="capture-card__label" style="color:${entry.color}">${entry.label}</span>
            <span class="capture-card__confidence">${(entry.confidence*100).toFixed(1)}%</span>
            <span class="capture-card__time">${entry.time}</span>`;
        grid.appendChild(card);
    });

    const rem = Math.max(0, MAX_CAPTURES - captureBuffer.length);
    for (let i = 0; i < rem; i++) {
        const ph = document.createElement('div');
        ph.className = 'capture-skeleton';
        grid.appendChild(ph);
    }
}