// URL backend Render
const API_URL = 'https://thisnumberdoesnotexit.onrender.com/';

// ============================================================
// Translation Data
// ============================================================
const translations = {
    en: {
        app_title: 'MNIST Generator | Neural Dark',
        h2_config: 'Configuration',
        lbl_mode: 'GENERATION MODE',
        mode_global_label: 'GLOBAL',
        mode_conditional_label: 'CONDITIONAL',
        lbl_digit: 'DIGIT SELECTION',
        lbl_batch: 'BATCH SIZE',
        lbl_filter: 'FILTERING',
        lbl_cleaning: 'Image Cleaning (Binarization)',
        lbl_cleaning_level: 'CLEANING LEVEL',
        cleaning_light: 'Light',
        cleaning_medium: 'Medium',
        cleaning_aggressive: 'Aggressive',
        lbl_rejection: 'Rejection Sampling',
        lbl_quality: 'QUALITY THRESHOLD',
        btn_generate: 'INITIALIZE SEQUENCE',
        status_ready: 'System ready.',
        status_working: 'Computing...',
        status_complete: count => `Generation complete (${count} images)`,
        status_error: 'System Error',
        info_tag_mode: 'MODE',
        info_tag_global_val: 'GLOBAL',
        info_tag_conditional_val: 'CONDITIONAL',
        gallery_placeholder: 'NO DATA GENERATED',
        error_no_images: 'Server returned no images.',
        error_fetch: 'Failed to connect to generator service.',
        info_tag_rate: 'RATE',
        tooltip_quality:
            'Lower percentile means stricter filtering (higher quality). Example: 10% keeps only the top 10% best samples.',
        tooltip_cleaning:
            'Applies a binarization filter (Light, Medium, or Aggressive thresholding) to improve image clarity.'
    },

    fr: {
        app_title: 'Générateur MNIST | Neural Dark',
        h2_config: 'Configuration',
        lbl_mode: 'MODE DE GÉNÉRATION',
        mode_global_label: 'GLOBAL',
        mode_conditional_label: 'CIBLÉ',
        lbl_digit: 'SÉLECTION DU CHIFFRE',
        lbl_batch: 'TAILLE DU BATCH',
        lbl_filter: 'FILTRAGE',
        lbl_cleaning: 'Nettoyage Image (Binarisation)',
        lbl_cleaning_level: 'NIVEAU DE NETTOYAGE',
        cleaning_light: 'Léger',
        cleaning_medium: 'Moyen',
        cleaning_aggressive: 'Agressif',
        lbl_rejection: 'Rejection Sampling',
        lbl_quality: 'SEUIL DE QUALITÉ',
        btn_generate: 'INITIALISER SÉQUENCE',
        status_ready: 'Système prêt.',
        status_working: 'Calcul en cours...',
        status_complete: count => `Génération terminée (${count} images)`,
        status_error: 'Erreur système',
        info_tag_mode: 'MODE',
        info_tag_global_val: 'GLOBAL',
        info_tag_conditional_val: 'CIBLÉ',
        gallery_placeholder: 'AUCUNE DONNÉE GÉNÉRÉE',
        error_no_images: 'Le serveur n’a renvoyé aucune image.',
        error_fetch: 'Échec de la connexion au service.',
        info_tag_rate: 'TAUX',
        tooltip_quality:
            'Un percentile plus petit signifie un filtrage plus strict (meilleure qualité).',
        tooltip_cleaning:
            'Applique un filtre de binarisation (Léger, Moyen ou Agressif).'
    }
};

// ============================================================
// Global State
// ============================================================
const body = document.body;

// Safe reading of data attributes injected by Flask
const HAS_GLOBAL = body.hasAttribute('data-has-global') && body.getAttribute('data-has-global') === 'true';
const HAS_CONDITIONAL = body.hasAttribute('data-has-conditional') && body.getAttribute('data-has-conditional') === 'true';

// Initial mode selection
let currentMode = 'global';
if (HAS_CONDITIONAL && !HAS_GLOBAL) currentMode = 'conditional';

let currentLang = 'en';
let currentDigit = 0;
let currentPercentile = 25;
let currentCleaningLevel = 'medium';

// ============================================================
// UI Elements
// ============================================================
const slider = document.getElementById('inp-samples');
const labelSamples = document.getElementById('lbl-samples');
const rejectionCheck = document.getElementById('inp-rejection');
const cleaningCheck = document.getElementById('inp-cleaning');
const pGroup = document.getElementById('percentile-group');
const cleaningOptionsDiv = document.getElementById('cleaning-options');
const gallery = document.getElementById('gallery');
const btnGen = document.getElementById('btn-generate');
const statusText = document.getElementById('status-text');
const errorBox = document.getElementById('error-display');
const infoTag = document.getElementById('info-tag');
const digitWrapper = document.getElementById('digit-wrapper');

// ============================================================
// Translation Helpers
// ============================================================
function T(key, args = []) {
    const value = translations[currentLang][key] ?? translations.en[key];
    return typeof value === 'function' ? value(...args) : value;
}

function updateCleaningButtonText() {
    document.querySelectorAll('.cleaning-btn').forEach(btn => {
        const method = btn.dataset.method;
        btn.textContent = T(`cleaning_${method}`);
    });
}

// ============================================================
// Language Handling
// ============================================================
function setLang(lang) {
    if (!translations[lang]) return;
    currentLang = lang;

    document.querySelectorAll('.lang-btn').forEach(b => b.classList.remove('active'));
    document.getElementById(`lang-${lang}`)?.classList.add('active');

    document.getElementById('app-title').textContent = T('app_title');
    document.getElementById('h2-config').textContent = T('h2_config');
    document.getElementById('lbl-mode').textContent = T('lbl_mode');
    document.getElementById('mode-global-label').textContent = T('mode_global_label');
    document.getElementById('mode-conditional-label').textContent = T('mode_conditional_label');
    document.getElementById('lbl-digit').textContent = T('lbl_digit');
    document.getElementById('lbl-batch').textContent = T('lbl_batch');
    document.getElementById('lbl-filter').textContent = T('lbl_filter');
    document.getElementById('lbl-cleaning').textContent = T('lbl_cleaning');
    document.getElementById('lbl-rejection').textContent = T('lbl_rejection');
    document.getElementById('lbl-quality').textContent = T('lbl_quality');
    document.getElementById('btn-generate').textContent = T('btn_generate');

    // SAFE label update (keeps tooltip)
    const cleaningLabel = document.getElementById('lbl-cleaning-level');
    if (cleaningLabel && cleaningLabel.childNodes.length > 0) {
        cleaningLabel.childNodes[0].nodeValue = T('lbl_cleaning_level') + ' ';
    }

    const tooltipQuality = document.getElementById('tooltip-quality');
    if (tooltipQuality) {
        tooltipQuality.textContent = T('tooltip_quality');
    }

    const tooltipCleaning = document.getElementById('tooltip-cleaning');
    if (tooltipCleaning) {
        tooltipCleaning.textContent = T('tooltip_cleaning');
    }

    const placeholder = document.getElementById('gallery-placeholder');
    if (placeholder) {
        placeholder.textContent = T('gallery_placeholder');
    }

    updateCleaningButtonText();
    statusText.textContent = T('status_ready');
    updateInfoTag();
}


// ============================================================
// UI Logic
// ============================================================
slider.addEventListener('input', e => {
    labelSamples.textContent = e.target.value;
});

function setMode(mode) {
    const btn = document.getElementById(`btn-mode-${mode}`);
    if (!btn) return; // safeguard: bouton non trouvé
    if (btn.classList.contains('disabled')) return; // bouton désactivé

    currentMode = mode;
    updateModeUI();
}

function updateModeUI() {
    document.querySelectorAll('.mode-btn').forEach(b => b.classList.remove('active'));

    const activeBtn = document.getElementById(`btn-mode-${currentMode}`);
    if (activeBtn) activeBtn.classList.add('active');

    const digitWrapper = document.getElementById('digit-wrapper');
    if (digitWrapper) {
        digitWrapper.style.display = (currentMode === 'conditional') ? 'block' : 'none';
    }

    const infoTag = document.getElementById('info-tag');
    if (infoTag) {
        let modeLabel = currentMode === 'global' ? 'GLOBAL' : 'CONDITIONAL';
        let infoString = `MODE: ${modeLabel}`;
        infoTag.textContent = infoString;
    }
}


function setDigit(digit, el) {
    currentDigit = digit;
    document.querySelectorAll('.digit-btn').forEach(b => b.classList.remove('active'));
    el.classList.add('active');
    updateInfoTag();
}

function setPercentile(val) {
    currentPercentile = val;
    document.querySelectorAll('.p-btn').forEach(btn => {
        btn.classList.toggle('active', btn.innerText.includes(val));
    });
}

function setCleaningLevel(level, el) {
    currentCleaningLevel = level;
    document.querySelectorAll('.cleaning-btn').forEach(b => b.classList.remove('active'));
    el.classList.add('active');
}

function toggleRejectionOptions(event, fromCheckbox = false) {
    if (event && event.target.tagName !== 'INPUT' && !fromCheckbox) {
        rejectionCheck.checked = !rejectionCheck.checked;
    }
    pGroup.style.display = rejectionCheck.checked ? 'block' : 'none';
}

function toggleCleaningOptions(event, fromCheckbox = false) {
    if (event && event.target.tagName !== 'INPUT' && !fromCheckbox) {
        cleaningCheck.checked = !cleaningCheck.checked;
    }
    cleaningOptionsDiv.style.display = cleaningCheck.checked ? 'block' : 'none';
}

function updateInfoTag() {
    let text = `${T('info_tag_mode')}: ` +
        (currentMode === 'global'
            ? T('info_tag_global_val')
            : T('info_tag_conditional_val'));

    if (currentMode === 'conditional') text += ` [${currentDigit}]`;
    infoTag.textContent = text;
}

// ============================================================
// Core Logic (API Simulation)
// ============================================================
async function generateImages() {
    btnGen.disabled = true;
    btnGen.innerHTML = '<span class="loader"></span>';
    statusText.textContent = T('status_working');
    statusText.style.color = 'var(--text-muted)';
    gallery.style.opacity = '0.5';
    errorBox.style.display = 'none';

    const payload = {
        n_samples: parseInt(slider.value),
        use_rejection: rejectionCheck.checked,
        percentile: currentPercentile,
        mode: currentMode,
        digit: currentMode === 'conditional' ? currentDigit : null,
        clean_images: cleaningCheck.checked,
        cleaning_method: currentCleaningLevel
    };

    try {
        const response = await fetch(`${API_URL}/generate`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(payload)
        });

        if (!response.ok) {
            throw new Error(`Server error (${response.status})`);
        }

        const data = await response.json();

        if (!data.success) {
            throw new Error(data.error || 'Generation failed');
        }

        if (!data.images || data.images.length === 0) {
            throw new Error(T('error_no_images'));
        }

        renderGallery(data.images);

        statusText.textContent = T('status_complete', [data.count]);
        statusText.style.color = 'var(--success)';

        let infoString = `${T('info_tag_mode')}: ${
            currentMode === 'global'
                ? T('info_tag_global_val')
                : T('info_tag_conditional_val')
        }`;

        if (currentMode === 'conditional') {
            infoString += ` [${data.digit}]`;
        }

        if (data.use_rejection && data.acceptance_rate !== undefined) {
            infoString += ` | ${T('info_tag_rate')}: ${data.acceptance_rate.toFixed(2)}`;
        }

        infoTag.textContent = infoString;

    } catch (err) {
        console.error(err);

        statusText.textContent = T('status_error');
        statusText.style.color = '#FF453A';

        errorBox.textContent = err.message;
        errorBox.style.display = 'block';

        gallery.innerHTML = `
            <div style="grid-column:1/-1;height:300px;display:flex;
                        align-items:center;justify-content:center;
                        border:1px dashed var(--border);
                        border-radius:var(--radius);">
                <span id="gallery-placeholder">${T('gallery_placeholder')}</span>
            </div>
        `;
    } finally {
        btnGen.disabled = false;
        btnGen.textContent = T('btn_generate');
        gallery.style.opacity = '1';
    }
}

function renderGallery(images) {
    gallery.innerHTML = '';
    images.forEach(src => {
        const div = document.createElement('div');
        div.className = 'img-card';
        div.innerHTML = `<img src="${src}" alt="Generated digit">`;
        gallery.appendChild(div);
    });
}

// ============================================================
// Initialization
// ============================================================
window.onload = () => {
    setLang(currentLang);

    const btnGlobal = document.getElementById('btn-mode-global');
    const btnConditional = document.getElementById('btn-mode-conditional');

    //if (!HAS_GLOBAL) btnGlobal.classList.add('disabled');
    //if (!HAS_CONDITIONAL) btnConditional.classList.add('disabled');

    updateModeUI();
    toggleRejectionOptions(null, false);
    cleaningOptionsDiv.style.display = cleaningCheck.checked ? 'block' : 'none';
    updateCleaningButtonText();
};
