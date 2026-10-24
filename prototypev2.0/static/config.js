// ─── CONFIG ───────────────────────────────────────────────
const CONFIG = {
    inferenceInterval : 250,
    adaptiveInterval  : true,
    minConfidence     : 0.50,
    streamWidth       : 640,
    streamHeight      : 480,
    captureWidth      : 224,
    captureHeight     : 224,
    jpegQuality       : 0.5
};

const emotionColors = {
    Happy   : '#39ffb4',
    Sad     : '#ff55aa',
    Fear    : '#aa55ff',
    Angry   : '#ff5555',
    Disgust : '#8DB600',
    Surprise: '#FF6B00',
    Neutral : '#55AAFF'
};

const EMOTION_ORDER     = ['Happy', 'Neutral', 'Sad', 'Fear', 'Angry', 'Disgust', 'Surprise'];
const NEGATIVE_EMOTIONS = ['Sad', 'Fear', 'Angry', 'Disgust'];

const AROUSAL_MAP = {
    Angry   : 0.95,
    Fear    : 0.90,
    Surprise: 0.80,
    Disgust : 0.70,
    Happy   : 0.65,
    Sad     : 0.35,
    Neutral : 0.20
};
const VALENCE_MAP = {
    Happy   :  0.85,
    Surprise:  0.30,
    Neutral :  0.00,
    Sad     : -0.60,
    Fear    : -0.70,
    Angry   : -0.80,
    Disgust : -0.75
};

const DISTANCE_ZONES = {
    tooClose : { max: 35,       label: 'Too Close', color: '#ff5555' },
    optimal  : { max: 80,       label: 'Optimal',   color: '#39ffb4' },
    far      : { max: 120,      label: 'Far',        color: '#FFD700' },
    tooFar   : { max: Infinity, label: 'Too Far',    color: '#ff5555' }
};

function getDistanceZone(cm) {
    if (!cm || cm <= 0) return null;
    if (cm < 35)        return DISTANCE_ZONES.tooClose;
    if (cm <= 80)       return DISTANCE_ZONES.optimal;
    if (cm <= 120)      return DISTANCE_ZONES.far;
    return DISTANCE_ZONES.tooFar;
}

// MediaPipe landmark indices (single source of truth)
const MP_IDX = {
    left_eye_outer  : 33,   left_eye_inner  : 133,
    left_eye_top    : 159,  left_eye_bottom : 145,
    right_eye_outer : 263,  right_eye_inner : 362,
    right_eye_top   : 386,  right_eye_bottom: 374,
    left_brow_peak  : 70,   right_brow_peak : 299,
    left_brow_inner : 107,  right_brow_inner: 336,
    left_brow_outer : 46,   right_brow_outer: 276,
    mouth_left      : 61,   mouth_right     : 291,
    mouth_top       : 0,    mouth_bottom    : 17,
    upper_lip_top   : 267,  lower_lip_bottom: 16,
    nose_tip        : 1,    chin            : 152,
    jaw_left        : 234,  jaw_right       : 454,
    left_cheek      : 116,  right_cheek     : 345,
    left_cheek2     : 123,  right_cheek2    : 352
};

// Shared axis style used by all Chart.js instances
const AXIS_STYLE = {
    ticks: { color: 'rgba(255,255,255,0.7)', font: { size: 10 } },
    grid : { color: 'rgba(255,255,255,0.1)' }
};

// ─── SHARED UTILITIES ─────────────────────────────────────
function setTextContent(id, value) {
    const el = document.getElementById(id);
    if (el) el.textContent = value;
}

function clamp(value, min, max) {
    return Math.min(Math.max(value, min), max);
}

function distance(a, b) {
    if (!a || !b) return 0;
    const dx = a.x - b.x;
    const dy = a.y - b.y;
    return Math.sqrt(dx * dx + dy * dy);
}

function getPoint(landmarks, index) {
    if (!landmarks || index < 0 || index >= landmarks.length) return null;
    const pt = landmarks[index];
    return Array.isArray(pt) ? { x: pt[0], y: pt[1] } : { x: pt.x, y: pt.y };
}

// Shared skeleton HTML (single source of truth)
function buildSkeletonHTML() {
    const widths = [45, 30, 60, 20, 75, 15, 50];
    return `<div class="emotion-bar-skeleton-list">
        ${widths.map(w => `
        <div class="emotion-bar-skeleton-row">
            <div class="ebar-skel-label"></div>
            <div class="ebar-skel-track">
                <div class="ebar-skel-fill" style="width:${w}%"></div>
            </div>
            <div class="ebar-skel-pct"></div>
        </div>`).join('')}
    </div>`;
}

// Shared chart update-or-create helper
function updateOrCreate(instance, buildFn, newLabels, newData) {
    if (instance) {
        instance.data.labels           = newLabels;
        instance.data.datasets[0].data = newData;
        instance.update('none');
        return instance;
    }
    return buildFn();
}