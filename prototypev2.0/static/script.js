const hamburgerBtn = document.getElementById('hamburger-btn');
const mobileNav = document.getElementById('mobile-nav');

const video = document.getElementById('video-feed-dash');
const canvasElements = {
    dashboard : document.getElementById('video-canvas-dash'),
    comparison : document.getElementById('video-canvas-comp')
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

const CONFIG = {
    inferenceInterval: 250, //ms
    minConfidence: .50,
    streamWidth: 640,
    streamHeight: 480,
    captureWidth: 320,
    captureHeight: 240,
    jpegQuality: .6
};

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
            lastResult ? (emotionColors[lastResult.emotion] || '#ffffff') : null,
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
                width: { ideal: CONFIG.streamWidth },
                height: { ideal: CONFIG.streamHeight },
                frameRate: { ideal: 30, max: 30 }
            }
        });
        videoStream = stream;

        // Set stream to the video element for frame capture
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
let isRunning = false;
let inferenceTimer = null;
let lastResult = null;
const captureCanvas = document.createElement('canvas');
captureCanvas.width = CONFIG.captureWidth;
captureCanvas.height = CONFIG.captureHeight;
const captureCtx = captureCanvas.getContext('2d');
const emotionColors = {
    happy: '#39ffb4', sad: '#ff55aa', fear: '#aa55ff',
    angry: '#ff5555', disgust: '#8DB600', surprise: '#FF6B00', neutral: '#55AAFF'
};

async function sendFrame() {
    if (!isRunning || !videoStream || video.readyState < 2) return;

    // Downscale frame to capture resolution before sending
    captureCtx.drawImage(video, 0, 0, CONFIG.captureWidth, CONFIG.captureHeight);

    const blob = await new Promise(resolve =>
        captureCanvas.toBlob(resolve, 'image/jpeg', CONFIG.jpegQuality)
    );

    const formData = new FormData();
    formData.append('frame', blob);

    try {
        const response = await fetch('/predict', { method: 'POST', body: formData });
        const result = await response.json();
        handleResult(result);
    } catch (err) {
        console.error('Inference error:', err);
    }
}

function handleResult(result) {
    lastResult = result;

    // Update UI readouts (dashboard only)
    const emotionDisplay = document.getElementById('emotion-display');
    const confidenceDisplay = document.getElementById('confidence-display');
    if (emotionDisplay) emotionDisplay.textContent = result.emotion;
    if (confidenceDisplay) confidenceDisplay.textContent = 
        (result.confidence * 100).toFixed(1) + '%';
}

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

    if (landmarks && landmarks.length) {
        context.fillStyle = colorHex;
        for (const [x, y] of landmarks) {
            context.beginPath();
            context.arc(x * scaleX, y * scaleY, 3, 0, 2 * Math.PI);
            context.fill();
        }
    }
}

function startInference() {
    if (isRunning) return;
    isRunning = true;
    inferenceTimer = setInterval(sendFrame, CONFIG.inferenceInterval);
}

function stopInference() {
    isRunning = false;
    clearInterval(inferenceTimer);
}

window.addEventListener('load', () => {
    // Start on Dashboard page as default
    canvas = canvasElements['dashboard'];
    ctx = canvas.getContext('2d');
    initCamera();

    document.querySelectorAll('.start-button').forEach(btn => {
        btn.addEventListener('click', () => {
            if (!isRunning) {
                startInference();
                btn.textContent = 'Stop';
            } else {
                stopInference();
                btn.textContent = 'Start Logging';
            }
        })
    });
});