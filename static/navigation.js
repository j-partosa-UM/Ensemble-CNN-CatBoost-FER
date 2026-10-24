// ─── NAVIGATION ───────────────────────────────────────────
const hamburgerBtn = document.getElementById('hamburger-btn');
const mobileNav    = document.getElementById('mobile-nav');

function switchPage(link) {
    document.querySelectorAll('.nav-link').forEach(el => el.classList.remove('active'));
    document.querySelectorAll('.page-section').forEach(el => el.classList.remove('active'));
    document.getElementById(link).classList.add('active');
    document.getElementById(link + '-page').classList.add('active');

    document.querySelectorAll('.mobile-nav-list .nav-link').forEach(el => {
        el.classList.toggle('active', el.id === `mob-${link}`);
    });

    activePage = link;

    document.querySelectorAll('.start-button').forEach(btn => {
        btn.textContent = link === 'comparison' ? 'Start Comparison' : 'Start Logging';
    });

    if (link === 'analysis') {
        requestAnimationFrame(() => renderChart());
        return;
    }

    if (link === 'model-info') {
        if (window._loadModelInfoMetrics) window._loadModelInfoMetrics();
        return
    }

    resetSessionCounters();
    setTextContent('comp-cnn-avg-conf', '--');
    setTextContent('comp-ens-avg-conf', '--');
    stopInference();

    if (canvasElements[link]) {
        canvas = canvasElements[link];
        initCanvas();
        ctx = canvas.getContext('2d');
    }
}

function closeMobileNav() {
    mobileNav.classList.remove('open');
    hamburgerBtn.classList.remove('open');
    hamburgerBtn.setAttribute('aria-expanded', false);
}

hamburgerBtn.addEventListener('click', () => {
    const isOpen = mobileNav.classList.toggle('open');
    hamburgerBtn.classList.toggle('open', isOpen);
    hamburgerBtn.setAttribute('aria-expanded', isOpen);
});

document.addEventListener('click', e => {
    if (!hamburgerBtn.contains(e.target) && !mobileNav.contains(e.target)) {
        closeMobileNav();
    }
});