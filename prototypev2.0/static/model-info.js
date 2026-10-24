// ─── MODEL INFO PAGE ──────────────────────────────────────
(function () {
    let metricsData = null;
    let activeModel = 'ensemble_cnn_catboost';
    let aucChart    = null;
    let f1Chart     = null;
 
    /* ── Fetch once and wire up ── */
    async function loadMetrics() {
        try {
            const res  = await fetch('/model_metrics');
            metricsData = await res.json();
            renderHero();
            renderCharts();
            renderTable(activeModel);
            renderCallouts();
        } catch (e) {
            console.error('Could not load model_metrics:', e);
        }
    }
 
    /* ── Hero numbers ── */
    function pct(v) { return (v * 100).toFixed(1) + '%'; }
 
    function renderHero() {
        const cnn = metricsData.models.cnn_resnet50.overall;
        const ens = metricsData.models.ensemble_cnn_catboost.overall;
        const imp = metricsData.improvements;
 
        document.querySelector('#hero-cnn-acc .hero-value').textContent = pct(cnn.accuracy);
        document.querySelector('#hero-ens-acc .hero-value').textContent = pct(ens.accuracy);
        document.querySelector('#hero-delta-acc').textContent           = `+${pct(imp.accuracy_delta)}`;
 
        document.querySelector('#hero-cnn-f1 .hero-value').textContent  = pct(cnn.macro_f1);
        document.querySelector('#hero-ens-f1 .hero-value').textContent  = pct(ens.macro_f1);
        document.querySelector('#hero-delta-f1').textContent            = `+${pct(imp.f1_delta)}`;
 
        document.querySelector('#hero-cnn-ece .hero-value').textContent = cnn.ece.toFixed(4);
        document.querySelector('#hero-ens-ece .hero-value').textContent = ens.ece.toFixed(4);
        const eceDelta = (imp.ece_delta * 100).toFixed(1);
        document.querySelector('#hero-delta-ece').textContent           = `${eceDelta}% (better)`;
        document.querySelector('#hero-delta-ece').classList.add('negative');
    }
 
    /* ── Charts ── */
    function renderCharts() {
        const cnn = metricsData.models.cnn_resnet50.per_class;
        const ens = metricsData.models.ensemble_cnn_catboost.per_class;
 
        /* Radar — AUC per emotion */
        const radarCtx = document.getElementById('auc-radar-chart').getContext('2d');
        if (aucChart) aucChart.destroy();
        aucChart = new Chart(radarCtx, {
            type: 'radar',
            data: {
                labels  : EMOTION_ORDER,
                datasets: [
                    {
                        label          : 'CNN',
                        data           : EMOTION_ORDER.map(e => +(cnn[e].auc * 100).toFixed(2)),
                        borderColor    : 'rgba(128,0,0,0.75)',
                        backgroundColor: 'rgba(128,0,0,0.08)',
                        pointBackgroundColor: 'rgba(128,0,0,0.85)',
                        borderWidth    : 2
                    },
                    {
                        label          : 'Ensemble',
                        data           : EMOTION_ORDER.map(e => +(ens[e].auc * 100).toFixed(2)),
                        borderColor    : 'rgba(255,170,0,0.9)',
                        backgroundColor: 'rgba(255,170,0,0.1)',
                        pointBackgroundColor: '#FFD700',
                        borderWidth    : 2
                    }
                ]
            },
            options: {
                responsive: true, maintainAspectRatio: false,
                scales: {
                    r: {
                        min: 85, max: 100,
                        ticks      : { color: 'rgba(0,0,0,0.35)', font: { size: 10 }, stepSize: 5, backdropColor: 'transparent' },
                        grid       : { color: 'rgba(0,0,0,0.08)' },
                        angleLines : { color: 'rgba(0,0,0,0.08)' },
                        pointLabels: { color: 'rgba(0,0,0,0.65)', font: { size: 11 } }
                    }
                },
                plugins: {
                    legend: { labels: { color: 'rgba(0,0,0,0.6)', font: { size: 11 }, boxWidth: 10 } }
                }
            }
        });
 
        /* Bar — F1 per emotion */
        const barCtx = document.getElementById('f1-bar-chart').getContext('2d');
        if (f1Chart) f1Chart.destroy();
        f1Chart = new Chart(barCtx, {
            type: 'bar',
            data: {
                labels  : EMOTION_ORDER,
                datasets: [
                    {
                        label          : 'CNN',
                        data           : EMOTION_ORDER.map(e => +(cnn[e].f1 * 100).toFixed(1)),
                        backgroundColor: 'rgba(128,0,0,0.5)',
                        borderColor    : 'rgba(128,0,0,0.85)',
                        borderWidth    : 1,
                        borderRadius   : 3
                    },
                    {
                        label          : 'Ensemble',
                        data           : EMOTION_ORDER.map(e => +(ens[e].f1 * 100).toFixed(1)),
                        backgroundColor: 'rgba(255,170,0,0.5)',
                        borderColor    : 'rgba(255,170,0,0.9)',
                        borderWidth    : 1,
                        borderRadius   : 3
                    }
                ]
            },
            options: {
                responsive: true, maintainAspectRatio: false,
                scales: {
                    x: {
                        ticks: { color: 'rgba(0,0,0,0.5)', font: { size: 10 } },
                        grid : { display: false }
                    },
                    y: {
                        min  : 50, max: 100,
                        ticks: { color: 'rgba(0,0,0,0.4)', font: { size: 10 }, callback: v => v + '%' },
                        grid : { color: 'rgba(0,0,0,0.06)' }
                    }
                },
                plugins: {
                    legend : { labels: { color: 'rgba(0,0,0,0.6)', font: { size: 11 }, boxWidth: 10 } },
                    tooltip: { callbacks: { label: ctx => `${ctx.dataset.label}: ${ctx.parsed.y}%` } }
                }
            }
        });
    }
 
    /* ── Per-class table ── */
    function renderTable(modelKey) {
        const model = metricsData.models[modelKey];
        const tbody = document.getElementById('metrics-table-body');
        tbody.innerHTML = '';
 
        EMOTION_ORDER.forEach(emo => {
            const d    = model.per_class[emo];
            const col  = emotionColors[emo] || '#800000';
            const barW = Math.round(d.f1 * 80);
            const tr   = document.createElement('tr');
            tr.innerHTML = `
                <td>
                    <span style="display:inline-flex;align-items:center;gap:0.45rem;">
                        <span style="width:8px;height:8px;border-radius:50%;background:${col};display:inline-block;flex-shrink:0;"></span>
                        ${emo}
                    </span>
                </td>
                <td>${(d.precision * 100).toFixed(1)}%</td>
                <td>${(d.recall    * 100).toFixed(1)}%</td>
                <td class="f1-cell">
                    <div class="f1-bar-bg">
                        <span>${(d.f1 * 100).toFixed(1)}%</span>
                        <div class="f1-bar-inline" style="width:${barW}px;background:${col};"></div>
                    </div>
                </td>
                <td>${d.support.toLocaleString()}</td>
                <td>${(d.auc * 100).toFixed(2)}%</td>
            `;
            tbody.appendChild(tr);
        });
 
        /* Macro-average footer row — light-theme colours */
        const o    = model.overall;
        const foot = document.createElement('tr');
        foot.style.borderTop = '2px solid rgba(128,0,0,0.12)';
        foot.innerHTML = `
            <td style="font-weight:600;color:rgba(0,0,0,0.4)">Macro avg</td>
            <td style="font-weight:600">${(o.macro_precision * 100).toFixed(1)}%</td>
            <td style="font-weight:600">${(o.macro_recall    * 100).toFixed(1)}%</td>
            <td style="font-weight:600">${(o.macro_f1        * 100).toFixed(1)}%</td>
            <td style="color:rgba(0,0,0,0.35)">—</td>
            <td style="font-weight:600">${(o.auc_macro       * 100).toFixed(2)}%</td>
        `;
        tbody.appendChild(foot);
    }
 
    /* ── Callouts ── */
    function renderCallouts() {
        const imp     = metricsData.improvements;
        const ds      = metricsData.dataset;
        const ens     = metricsData.models.ensemble_cnn_catboost;
        const confused = ens.confusion_most_confused_pair;
 
        document.getElementById('callout-confused-text').textContent =
            `The model most frequently confuses ${confused[0]} and ${confused[1]}. ` +
            `These share overlapping facial muscle movements in the upper face.`;
 
        const ecePct = Math.abs(imp.ece_delta / metricsData.models.cnn_resnet50.overall.ece * 100).toFixed(0);
        document.getElementById('callout-ece-text').textContent =
            `Ensemble reduces calibration error (ECE) by ~${ecePct}% relative to the CNN baseline, ` +
            `meaning confidence scores are more reliable.`;
 
        document.getElementById('callout-dataset-text').textContent =
            `${ds.name} — ${ds.total_samples.toLocaleString()} samples across 7 emotion classes.`;
    }
 
    /* ── Tab switching ── */
    document.querySelectorAll('.mi-tab').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.mi-tab').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            activeModel = btn.dataset.model;
            if (metricsData) renderTable(activeModel);
        });
    });
 
    /* ── Exposed hook for navigation.js ── */
    window._loadModelInfoMetrics = function () {
        if (!metricsData) loadMetrics();
    };
 
    /* ── Auto-load if page is already active on DOMContentLoaded ── */
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', () => {
            if (document.getElementById('model-info-page')?.classList.contains('active')) {
                loadMetrics();
            }
        });
    }
})();