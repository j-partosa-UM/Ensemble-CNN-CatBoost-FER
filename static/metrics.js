// ─── FACIAL METRICS + HISTORY ─────────────────────────────
const facialMetricHistory = [];
const earHistory          = [];
const auHistory           = [];
const arousalHistory      = [];
const distanceHistory     = []; 
const MAX_METRIC_HISTORY  = 30;
const MAX_HISTORY         = 60;

let blinkBelowThreshold = false;   

function detectBlink(ear) {       
    const THRESHOLD = 0.21;
    if (ear < THRESHOLD) {
        blinkBelowThreshold = true;
        return false;
    } else {
        if (blinkBelowThreshold) {
            blinkBelowThreshold = false;
            return true;
        }
        return false;
    }
}

function getSmileCategory(score) {
    if (score < 30) return 'Slight Smile';
    if (score < 55) return 'Open Smile';
    if (score < 80) return 'Intense Smile';
    return 'Grin';
}

function computeFacialMetrics(landmarks) {
    if (!landmarks || landmarks.length < 475) return null;

    const p = key => getPoint(landmarks, MP_IDX[key]);

    const leftEyeW = distance(p('left_eye_outer'), p('left_eye_inner'));
    const rightEyeW = distance(p('right_eye_outer'), p('right_eye_inner'));
    const leftEyeH = distance(p('left_eye_top'), p('left_eye_bottom'));
    const rightEyeH = distance(p('right_eye_top'), p('right_eye_bottom'));
    const mouthW = distance(p('mouth_left'), p('mouth_right'));
    const mouthH = distance(p('mouth_top'), p('mouth_bottom'));
    const faceW = distance(p('jaw_left'), p('jaw_right'));
    const faceH = distance(p('nose_tip'), p('chin'));

    if (faceW <= 0 || faceH <= 0) return null;

    // EAR — Soukupová & Čech formula
    const leftEAR = leftEyeH / Math.max(leftEyeW, 1);
    const rightEAR = rightEyeH / Math.max(rightEyeW, 1);
    const EAR = (leftEAR + rightEAR) / 2;

    const mouthOpen = mouthH / Math.max(faceH, 1);
    const lipCornerCurve = (((p('mouth_left')?.y ?? 0) + (p('mouth_right')?.y ?? 0)) / 2
        - (p('upper_lip_top')?.y ?? 0)) / Math.max(faceH, 1);
    const smileBase = (mouthW / Math.max(faceW, 1)) * 100;
    const smileScore = clamp(smileBase + (lipCornerCurve * 70) + (mouthOpen * 30), 0, 100);
    const lipStretch = clamp((mouthW / Math.max(faceW, 1)) * 120, 0, 100);
    const jawOpen = clamp(distance(p('chin'), p('lower_lip_bottom')) / Math.max(faceH, 1) * 120, 0, 100);
    const browRaise = clamp(
        ((distance(p('left_brow_peak'), p('left_eye_top')) +
            distance(p('right_brow_peak'), p('right_eye_top'))) / 2)
        / Math.max(faceH, 1) * 200, 0, 100
    );
    const browLower = clamp(
        100 - ((distance(p('left_brow_inner'), p('left_eye_top')) +
            distance(p('right_brow_inner'), p('right_eye_top'))) / 2)
        / Math.max(faceH, 1) * 250, 0, 100
    );

    // AU proxies (FACS-based)
    const au = {
        AU1: clamp(distance(p('left_brow_inner'), p('left_eye_top')) / Math.max(faceH, 1) * 300, 0, 100),
        AU2: clamp(distance(p('left_brow_outer'), p('left_eye_top')) / Math.max(faceH, 1) * 300, 0, 100),
        AU4: browLower,
        AU6: clamp(distance(p('left_cheek'), p('left_cheek2')) / Math.max(faceW, 1) * 250, 0, 100),
        AU12: clamp((mouthW / Math.max(faceW, 1)) * 150, 0, 100),
        AU17: clamp(distance(p('chin'), p('lower_lip_bottom')) / Math.max(faceH, 1) * 200, 0, 100),
        AU25: clamp(mouthOpen * 200, 0, 100)
    };

    return {
        EAR,
        isBlink: detectBlink(EAR),
        isDuchenne   : au.AU6 > 25 && au.AU12 > 35,
        eyeContact   : clamp(EAR * 220, 0, 100),
        smileScore,
        smileCategory: getSmileCategory(smileScore),
        micro: [
            Math.round(browRaise),
            Math.round(clamp(EAR * 100, 0, 100)),
            Math.round(clamp(mouthOpen * 150, 0, 100)),
            Math.round(lipStretch),
            Math.round(jawOpen)
        ],
        au
    };
}

function appendFacialMetrics(metrics) {
    if (!metrics) return;

    facialMetricHistory.push(metrics);
    if (facialMetricHistory.length > MAX_METRIC_HISTORY) facialMetricHistory.shift();

    earHistory.push({ ear: metrics.EAR, blink: metrics.isBlink });
    if (earHistory.length > MAX_HISTORY) earHistory.shift();

    auHistory.push(metrics.au);
    if (auHistory.length > MAX_HISTORY) auHistory.shift();
}

function appendDistanceMetric(cm) {
    if (cm == null) return;
    const zone = getDistanceZone(cm);
    distanceHistory.push({ cm, zone });
    if (distanceHistory.length > MAX_HISTORY) distanceHistory.shift();
}

function clearMetricHistory() {
    facialMetricHistory.length = 0;
    earHistory.length          = 0;
    auHistory.length           = 0;
    arousalHistory.length      = 0;
    distanceHistory.length     = 0;
    blinkBelowThreshold        = false;
}
