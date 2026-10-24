// ─── MAIN — entry point ───────────────────────────────────
const video = document.getElementById('video-feed-dash');
const canvasElements = {
    dashboard : document.getElementById('video-canvas-dash'),
    comparison: document.getElementById('video-canvas-comp')
};
let canvas      = null;
let ctx         = null;
let videoStream = null;

function initCanvas() {
    if (!videoStream) return;
    canvas.width  = videoStream.getVideoTracks()[0].getSettings().width  || CONFIG.streamWidth;
    canvas.height = videoStream.getVideoTracks()[0].getSettings().height || CONFIG.streamHeight;
}

function drawLoop() {
    if (canvas && ctx && video.readyState >= 2) {
        drawFrame(
            video, ctx, canvas,
            lastResult?.bbox  || null,
            lastResult ? (emotionColors[lastResult.label] || '#fff') : null,
            lastResult?.landmarks || null
        );
    }
    requestAnimationFrame(drawLoop);
}

function drawFrame(video, context, canvas, bbox=null, colorHex=null, landmarks=null) {
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    const scaleX = canvas.width  / CONFIG.captureWidth;
    const scaleY = canvas.height / CONFIG.captureHeight;

    if (bbox?.length === 4) {
        const [x, y, w, h] = bbox;
        context.strokeStyle = colorHex;
        context.lineWidth   = 4;
        context.strokeRect(x*scaleX, y*scaleY, w*scaleX, h*scaleY);
    }

    if (landmarks?.length) {
        context.globalAlpha = 0.6;
        context.fillStyle   = colorHex;
        for (const [x, y] of landmarks) {
            context.beginPath();
            context.arc(x*scaleX, y*scaleY, 1.5, 0, 2*Math.PI);
            context.fill();
        }
        context.globalAlpha = 1.0;
    }
}

async function initCamera() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: {
                width    : { ideal: CONFIG.streamWidth },
                height   : { ideal: CONFIG.streamHeight },
                frameRate: { ideal: 30, max: 30 }
            }
        });
        videoStream    = stream;
        video.srcObject = stream;
        video.onloadedmetadata = () => { initCanvas(); drawLoop(); };
    } catch (err) {
        alert('Camera error: ' + err.message);
    }
}

window.addEventListener('load', () => {
    activePage = 'dashboard';
    canvas     = canvasElements['dashboard'];
    ctx        = canvas.getContext('2d');
    initCamera();

    document.querySelectorAll('.start-button').forEach(btn => {
        btn.addEventListener('click', () => {
            if (!isRunning) {
                activePage = btn.closest('#comparison-page') ? 'comparison' : 'dashboard';
                canvas     = canvasElements[activePage] || canvasElements['dashboard'];
                ctx        = canvas.getContext('2d');
                initCanvas();
                startInference();
                btn.textContent = activePage === 'comparison' ? 'Stop Comparison' : 'Stop';
            } else {
                stopInference();
                btn.textContent = activePage === 'comparison' ? 'Start Comparison' : 'Start Logging';
            }
        });
    });
});
