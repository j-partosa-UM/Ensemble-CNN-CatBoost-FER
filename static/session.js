// ─── SESSION LOG (linked list) ────────────────────────────
const MAX_LOG_ENTRIES = 100;

class LogNode {
    constructor(data, next = null) {
        this.data = data;
        this.next = next;
    }
}

class SessionLog {
    constructor() { this.head = null; this.size = 0; }
    prepend(data) { this.head = new LogNode(data, this.head); this.size++; }
}

const sessionLog = new SessionLog();
let placeholderRemoved = false;

function appendSessionLog(label, confidence, inferenceMs) {
    const logList = document.getElementById('log-list');
    if (!logList) return;

    if (!placeholderRemoved) {
        logList.querySelector('[data-placeholder]')?.remove();
        placeholderRemoved = true;
    }

    const time = new Date().toLocaleTimeString([], {
        hour: '2-digit', minute: '2-digit', second: '2-digit'
    });

    const entry = document.createElement('div');
    entry.className = 'session-log-list-item';
    entry.style.borderLeftColor = emotionColors[label];
    entry.style.background      = 'rgba(255,255,255,0.03)';
    entry.innerHTML = `
        <span class="log-time">${time}</span>
        <span class="log-emotion" style="color:${emotionColors[label]||'#fff'}">${label}</span>
        <span class="log-confidence">${(confidence * 100).toFixed(1)}%</span>
        <span class="log-inference">${inferenceMs} ms</span>`;

    logList.insertBefore(entry, logList.firstChild);
    sessionLog.prepend({ timestamp: time, label, confidence });

    while (logList.children.length > MAX_LOG_ENTRIES) {
        logList.removeChild(logList.lastChild);
    }
}

// Aggregate session stats — used by export.js and charts.js
function aggregateSessionStats() {
    const counts    = {};
    const confByEmo = {};
    let   totalConf = 0;
    let   node      = sessionLog.head;

    while (node) {
        const { label, confidence } = node.data;
        counts[label]    = (counts[label] || 0) + 1;
        confByEmo[label] = confByEmo[label] || [];
        confByEmo[label].push(confidence);
        totalConf += confidence;
        node = node.next;
    }

    return { counts, confByEmo, total: sessionLog.size, totalConf };
}

// Build chronological entries array from linked list
function getChronologicalEntries() {
    const entries = [];
    let node = sessionLog.head;
    while (node) { entries.unshift(node.data); node = node.next; }
    return entries;
}