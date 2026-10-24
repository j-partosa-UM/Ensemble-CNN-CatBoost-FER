// ─── CHART INSTANCES ──────────────────────────────────────
let emotionChartInstance   = null;
let earChartInstance       = null;
let auChartInstance        = null;
let stabilityChartInstance = null;

// ─── SKELETON TOGGLE ──────────────────────────────────────
function toggleSkeletonLoaders(show) {
    ['emotion-skeleton','gantt-skeleton','arousal-skeleton',
     'negaffect-skeleton','ear-skeleton','au-skeleton',
     'stability-skeleton','distance-skeleton']
        .forEach(id => document.getElementById(id)?.classList.toggle('visible', show));
}

// ─── GAUGE HELPER ─────────────────────────────────────────
function setGaugeFill(fillId, pct, valueId, valueText) {
    const fill = document.getElementById(fillId);
    if (fill) fill.style.width = `${clamp(pct, 0, 100)}%`;
    if (valueId) setTextContent(valueId, valueText);
}

// ─── MAIN RENDER ──────────────────────────────────────────
function renderChart() {
    if (sessionLog.size === 0) {
        [emotionChartInstance, earChartInstance, auChartInstance, stabilityChartInstance]
            .forEach(c => c?.destroy());
        emotionChartInstance = earChartInstance =
        auChartInstance = stabilityChartInstance = null;
        toggleSkeletonLoaders(true);
        if (activePage === 'analysis' && isRunning)
            setTimeout(() => requestAnimationFrame(renderChart), 500);
        return;
    }

    toggleSkeletonLoaders(false);

    const { counts } = aggregateSessionStats();
    const entries    = getChronologicalEntries();
    const total      = sessionLog.size;

    // ══════════════════════════════════════════════════════
    // 1. Facial Expression Distribution — doughnut
    // ══════════════════════════════════════════════════════
    const allEmotionsOrder = ['Happy','Sad','Fear','Angry','Disgust','Surprise','Neutral'];
    const distLabels = Object.keys(counts);
    const distData   = Object.values(counts);
    const distColors = distLabels.map(l => emotionColors[l] || '#ccc');
    const distTotal  = distData.reduce((sum, v) => sum + v, 0);
    const distPcts   = distData.map(v => ((v / distTotal) * 100).toFixed(1));

    emotionChartInstance = updateOrCreate(
        emotionChartInstance,
        () => new Chart(document.getElementById('emotionChart').getContext('2d'), {
            type: 'doughnut',
            data: {
                labels  : distLabels.map(l => l.toUpperCase()),
                datasets: [{ data: distData, backgroundColor: distColors,
                             borderWidth: 0, hoverOffset: 6 }]
            },
            options: {
                responsive: true, maintainAspectRatio: false,
                cutout: '68%',
                plugins: {
                    legend: {
                        display  : true,
                        position : 'bottom',
                        labels   : {
                            color        : 'rgba(255, 255, 255, 0.75)',
                            font         : { size: 11 },
                            pointStyle   : 'circle',
                            usePointStyle: true,
                            padding      : 12,
                            boxWidth     : 10,
                            boxHeight    : 10,
                            generateLabels: chart => {
                                // ← always show all 7 emotions, fallback to --
                                return allEmotionsOrder.map((emotion, i) => {
                                    const count = counts[emotion] || 0;
                                    const pct   = count > 0 ? ((count / distTotal) * 100).toFixed(1) : null;
                                    return {
                                        text       : `${emotion.toUpperCase()}  ${pct ? pct + '%' : '--'}`,
                                        fillStyle  : emotionColors[emotion] || '#ccc',
                                        strokeStyle: 'transparent',
                                        fontColor  : 'rgba(255,255,255,0.75)',
                                        color      : 'rgba(255,255,255,0.75)',
                                        pointStyle : 'circle',
                                        index      : i,
                                    };
                                });
                            }
                        }
                    },
                    tooltip: { callbacks: {
                        label: ctx => ` ${ctx.label}: ${distPcts[ctx.dataIndex]}% (${ctx.parsed})`
                    }}
                }
            }
        }),
        distLabels.map(l => l.toUpperCase()), distData
    );
    if (emotionChartInstance) {
        emotionChartInstance.data.datasets[0].backgroundColor = distColors;
        emotionChartInstance.options.plugins.legend.labels.color = 'rgb(255, 255, 255)';
        emotionChartInstance.update('none');
    }

    const elapsedSec = sessionStartTime ? ((Date.now() - sessionStartTime) / 1000).toFixed(0) : (total * 0.25).toFixed(0);
    setTextContent('session-duration', `${elapsedSec}s`);

    // dominant expression — center label + meta
    const [domLabel, domCount] = Object.entries(counts).sort((a, b) => b[1] - a[1])[0];
    const domPct = (domCount / distTotal) * 100;
    setTextContent('dominant-emotion-label', domLabel);
    setTextContent('dominant-emotion-pct',   `${domLabel} — ${domPct.toFixed(1)}%`);

    // ══════════════════════════════════════════════════════
    // 2. Expression TIMELINE — Gantt (DOM)
    // ══════════════════════════════════════════════════════
    const ganttTrackEl  = document.getElementById('ganttTrack');
    const ganttTimesEl  = document.getElementById('ganttTimestamps');
    const ganttLegendEl = document.getElementById('ganttLegend');

    if (ganttTrackEl && entries.length > 0) {
        const segments = [];
        entries.forEach((e, i) => {
            if (i === 0 || e.label !== entries[i - 1].label)
                segments.push({ label: e.label, start: i, end: i, time: e.timestamp });
            else
                segments[segments.length - 1].end = i;
        });

        setTextContent('emotion-shifts',    segments.length - 1);
        setTextContent('timeline-dominant', `${domLabel} (${domPct.toFixed(0)}%)`);

        ganttTrackEl.innerHTML = segments.map(seg => {
            const w = ((seg.end - seg.start + 1) / entries.length * 100).toFixed(2); // ← fix
            const c = emotionColors[seg.label] || '#ccc';
            return `<div class="gantt-segment" style="width:${w}%;background:${c};"
                title="${seg.label} — ${seg.time} (${seg.end - seg.start + 1} frames)">
                ${parseFloat(w) > 8 ? `<span class="gantt-seg-label">${seg.label}</span>` : ''}
            </div>`;
        }).join('');

        // evenly sampled timestamps
        if (ganttTimesEl) {
            const sampleCount = Math.min(6, entries.length);
            ganttTimesEl.innerHTML = Array.from({ length: sampleCount }, (_, i) => {
                const idx = sampleCount === 1 ? 0
                    : Math.round(i * (entries.length - 1) / (sampleCount - 1));
                return `<span>${entries[idx].timestamp}</span>`;
            }).join('');
        }

        // legend — most recent unique emotions
        if (ganttLegendEl) {
            const seen = new Set();
            const recent = [];
            for (let i = segments.length - 1; i >= 0 && recent.length < 5; i--) {
                if (!seen.has(segments[i].label)) {
                    seen.add(segments[i].label);
                    recent.push(segments[i]);
                }
            }
            ganttLegendEl.innerHTML = recent.map(seg => `
                <span class="gantt-legend-item">
                    <span class="gantt-dot" style="background:${emotionColors[seg.label]||'#ccc'}"></span>
                    ${seg.label} <small>${seg.time}</small>
                </span>`).join('');
        }
    }

    // ══════════════════════════════════════════════════════
    // 3. AROUSAL & VALENCE — dual gauge (DOM)
    // ══════════════════════════════════════════════════════
    const latestLabel = entries[entries.length - 1]?.label;
    const arousalRaw  = arousalHistory.length > 0
        ? arousalHistory[arousalHistory.length - 1]
        : (AROUSAL_MAP[latestLabel] ?? 0.5);
    const valenceRaw  = VALENCE_MAP[latestLabel] ?? 0;

    const arousalPct = Math.round(arousalRaw * 100);
    const valencePct = Math.round((valenceRaw + 1) / 2 * 100);

    const arousalLabel = arousalRaw > 0.7 ? 'High — activated'
                       : arousalRaw > 0.4 ? 'Moderate' : 'Low — calm';
    const valenceLabel = valenceRaw >  0.2 ? 'Positive'
                       : valenceRaw < -0.2 ? 'Negative' : 'Neutral';

    setGaugeFill('arousal-fill', arousalPct, 'arousal-value', `${arousalPct}%`);
    setGaugeFill('valence-fill', valencePct, 'valence-value',
        `${valenceRaw >= 0 ? '+' : ''}${valenceRaw.toFixed(2)}`);

    setTextContent('arousal-level', arousalLabel);
    setTextContent('valence-level', valenceLabel);

    // combined state badge
    const avBadge = document.getElementById('av-state-badge');
    if (avBadge) {
        const state = arousalRaw > 0.6 && valenceRaw < -0.2 ? 'Distressed'
                    : arousalRaw > 0.6 && valenceRaw >  0.2 ? 'Excited'
                    : arousalRaw < 0.4 && valenceRaw >  0.2 ? 'Relaxed'
                    : arousalRaw < 0.4 && valenceRaw < -0.2 ? 'Depressed'
                    : 'Neutral';
        avBadge.textContent = state;
        avBadge.style.color = arousalRaw > 0.6 && valenceRaw < -0.2 ? '#ff5555'
                            : arousalRaw > 0.6 && valenceRaw >  0.2 ? '#39ffb4'
                            : 'rgba(255,255,255,0.6)';
    }

    // ══════════════════════════════════════════════════════
    // 4. NEGATIVE AFFECT LOAD — gauges (DOM)
    // ══════════════════════════════════════════════════════
    const negPctOf = emo => distTotal > 0 ? (counts[emo] || 0) / distTotal * 100 : 0; // ← fix
    const negPcts  = {
        Angry  : negPctOf('Angry'),
        Sad    : negPctOf('Sad'),
        Fear   : negPctOf('Fear'),
        Disgust: negPctOf('Disgust'),
    };
    const negTotal = Object.values(negPcts).reduce((a, b) => a + b, 0);

    setGaugeFill('neg-affect-fill-anger',   negPcts.Angry,   'neg-affect-anger',   `${negPcts.Angry.toFixed(1)}%`);
    setGaugeFill('neg-affect-fill-sad',     negPcts.Sad,     'neg-affect-sad',     `${negPcts.Sad.toFixed(1)}%`);
    setGaugeFill('neg-affect-fill-fear',    negPcts.Fear,    'neg-affect-fear',    `${negPcts.Fear.toFixed(1)}%`);
    setGaugeFill('neg-affect-fill-disgust', negPcts.Disgust, 'neg-affect-disgust', `${negPcts.Disgust.toFixed(1)}%`);
    setGaugeFill('neg-affect-fill-total',   negTotal,        'neg-affect-total',   `${negTotal.toFixed(1)}%`);
    setTextContent('neg-affect-pct', `${negTotal.toFixed(1)}%`);

    const totalFillEl = document.getElementById('neg-affect-fill-total');
    if (totalFillEl) {
        totalFillEl.style.background = negTotal >= 40
            ? 'linear-gradient(90deg,#FFD700,#ff5555)'
            : 'linear-gradient(90deg,#39ffb4,#FFD700)';
    }

    // ══════════════════════════════════════════════════════
    // 5. EAR + BLINKS — line chart
    // ══════════════════════════════════════════════════════
    if (earHistory.length > 0) {
        const earLabels  = earHistory.map((_, i) => `#${i + 1}`);
        const earData    = earHistory.map(e => parseFloat((e.ear * 100).toFixed(1)));
        // blink overlay — null for non-blinks so the line dataset shows a dot only on blink frames
        const blinkData  = earHistory.map((e, i) => e.blink ? earData[i] : null);

        setTextContent('blink-count', earHistory.filter(e => e.blink).length);
        setTextContent('avg-ear',
            (earHistory.reduce((s, e) => s + e.ear, 0) / earHistory.length).toFixed(3));

        if (earChartInstance) {
            earChartInstance.data.labels                       = earLabels;
            earChartInstance.data.datasets[0].data             = earData;
            earChartInstance.data.datasets[1].data             = blinkData;
            earChartInstance.data.datasets[1].pointRadius      = earHistory.map(e => e.blink ? 6 : 0);
            earChartInstance.update('none');
        } else {
            earChartInstance = new Chart(
                document.getElementById('earChart').getContext('2d'), {
                type: 'line',
                data: { labels: earLabels, datasets: [
                    { label:'EAR', data: earData,
                      borderColor:'#39ffb4', backgroundColor:'rgba(57,255,180,0.1)',
                      borderWidth:2, fill:true, tension:0.4,
                      pointRadius:0, pointHoverRadius:3 },
                    { label:'Blink', data: blinkData,
                      borderColor:'transparent', backgroundColor:'#FFD700',
                      pointRadius: earHistory.map(e => e.blink ? 6 : 0),
                      pointStyle:'triangle', pointHoverRadius:7,
                      spanGaps:false, showLine:false }
                ]},
                options: {
                    responsive:true, maintainAspectRatio:false,
                    plugins: { legend:{ display:true, position:'top',
                                        labels:{ color:'white', font:{ size:10 },
                                                 filter: item => item.text !== '' } } },
                    scales: {
                        y: { beginAtZero:true, max:100, ...AXIS_STYLE },
                        x: { ...AXIS_STYLE, ticks:{ ...AXIS_STYLE.ticks, maxTicksLimit:8 } }
                    }
                }
            });
        }
    } else {
        setTextContent('blink-count', '0');
        setTextContent('avg-ear', '--');
    }

    // ══════════════════════════════════════════════════════
    // 6. FACIAL ACTION UNITS — radar
    // ══════════════════════════════════════════════════════
    if (auHistory.length > 0) {
        const auKeys   = ['AU1','AU2','AU4','AU6','AU10','AU12','AU17'];
        const auLabels = ['AU1 Inner Brow','AU2 Outer Brow','AU4 Brow Lower',
                          'AU6 Cheek Raise','AU10 Upper Lip','AU12 Lip Corner','AU17 Chin Raise'];
        const auData   = auKeys.map(k => parseFloat(
            (auHistory.reduce((s,f) => s+(f[k]||0), 0) / auHistory.length).toFixed(1)
        ));
        const latest   = auHistory[auHistory.length - 1];
        const lid      = (latest['AU6']  || 0);
        const mc       = (latest['AU12'] || 0);
        const peakIdx = auData.indexOf(Math.max(...auData));
        setTextContent('peak-au', `${auLabels[peakIdx]} (${auData[peakIdx].toFixed(1)})`);
        setTextContent('genuine-smile', (lid > 25 && mc > 35) ? ' Duchenne' : ' Non-Duchenne');

        auChartInstance = updateOrCreate(
            auChartInstance,
            () => new Chart(document.getElementById('auChart').getContext('2d'), {
                type: 'radar',
                data: { labels: auLabels, datasets: [
                    { label:'AU Score', data: auData,
                      borderColor:'#FFD700', backgroundColor:'rgba(255,215,0,0.15)',
                      borderWidth:2, pointBackgroundColor:'#FFD700',
                      pointBorderColor:'#fff', pointRadius:4 }
                ]},
                options: { responsive:true, maintainAspectRatio:false,
                    plugins: { legend:{ display:false } },
                    scales: { r: {
                        beginAtZero:true, max:100,
                        ticks      : { stepSize:10, color:'rgba(255,255,255,0.3)',
                                       font:{size:9}, backdropColor:'transparent' },
                        grid       : { color:'rgba(255,255,255,0.1)' },
                        angleLines : { color:'rgba(255,255,255,0.15)' },
                        pointLabels: { color:'rgba(255,255,255,0.9)',
                                       font:{size:9,weight:'500',family:'Poppins'},
                                       backdropColor:'rgba(128,0,0,0.75)',
                                       backdropPadding:3, padding:6 }
                    }}
                }
            }),
            auLabels, auData
        );
        if (auChartInstance) {
            auChartInstance.data.datasets[0].data = auData;
            auChartInstance.update('none');
        }
    }

    // ══════════════════════════════════════════════════════
    // 7. EMOTIONAL STABILITY — line chart
    // ══════════════════════════════════════════════════════
    const WINDOW     = 10;
    const stabData   = [];
    const stabLabels = [];
    for (let i = WINDOW; i <= entries.length; i += WINDOW) {
        const win = entries.slice(i - WINDOW, i);
        let changes = 0;
        for (let j = 1; j < win.length; j++)
            if (win[j].label !== win[j - 1].label) changes++;
        stabData.push(parseFloat(((1 - changes / (WINDOW - 1)) * 100).toFixed(1)));
        stabLabels.push(win[win.length - 1].timestamp);
    }

    if (stabData.length > 0) {
        const cur   = stabData[stabData.length - 1];
        const state = cur > 70 ? 'Regulated' : cur > 40 ? 'Labile' : 'Dysregulated';
        setTextContent('stability-index', `${cur}%`);
        setTextContent('stability-state', state);

        stabilityChartInstance = updateOrCreate(
            stabilityChartInstance,
            () => new Chart(document.getElementById('stabilityChart').getContext('2d'), {
                type: 'line',
                data: { labels: stabLabels, datasets: [{
                    label:'Stability %', data: stabData,
                    borderColor:'#aa55ff', backgroundColor:'rgba(170,85,255,0.15)',
                    borderWidth:2, fill:true, tension:0.4,
                    pointBackgroundColor:'#aa55ff', pointRadius:4
                }]},
                options: {
                    responsive:true, maintainAspectRatio:false,
                    plugins: { legend:{ display:false } },
                    scales: {
                        y: { beginAtZero:true, max:100, ...AXIS_STYLE,
                             ticks:{ ...AXIS_STYLE.ticks, callback: v => `${v}%` } },
                        x: { ...AXIS_STYLE }
                    }
                }
            }),
            stabLabels, stabData
        );
        if (stabilityChartInstance) {                                       
            stabilityChartInstance.data.labels           = stabLabels;
            stabilityChartInstance.data.datasets[0].data = stabData;
            stabilityChartInstance.update('none');
        }  
    }

    // ── schedule next render ───────────────────────────────
    if (activePage === 'analysis' && isRunning)
        setTimeout(() => requestAnimationFrame(renderChart), 500);
}