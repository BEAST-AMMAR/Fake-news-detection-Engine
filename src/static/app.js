/* ==========================================================================
   Tribrid News Credibility Dashboard - Client Logic
   ========================================================================== */

document.addEventListener('DOMContentLoaded', () => {
    // Elements
    const tabUrlBtn = document.getElementById('tab-url-btn');
    const tabTextBtn = document.getElementById('tab-text-btn');
    const tabUrlContent = document.getElementById('tab-url-content');
    const tabTextContent = document.getElementById('tab-text-content');
    
    const inputUrl = document.getElementById('input-url');
    const inputTitle = document.getElementById('input-title');
    const inputText = document.getElementById('input-text');
    const btnVerify = document.getElementById('btn-verify');
    const btnText = document.getElementById('btn-text');
    const spinner = document.getElementById('spinner');

    const resultsWrapper = document.getElementById('results-wrapper');
    const gaugeProgress = document.getElementById('gauge-progress');
    const gaugeValue = document.getElementById('gauge-value');
    const verdictBadge = document.getElementById('verdict-badge');
    const confidenceTag = document.getElementById('confidence-tag');
    const latencyTag = document.getElementById('latency-tag');
    const articleHeadline = document.getElementById('article-headline');
    const primaryReason = document.getElementById('primary-reason');

    let currentMode = 'url';

    // Preset Data
    const PRESETS = {
        reuters: {
            title: "Global central banks hold benchmark rates steady as inflation cools",
            text: "LONDON (Reuters) - Major central banks maintained benchmark interest rates on Thursday as consumer price data showed continued deceleration across key economies. Officials indicated future policy decisions would remain data-dependent while monitoring employment figures."
        },
        onion: {
            url: "https://theonion.com/man-who-thought-hed-lost-all-hope-loses-last-additional-bit-of-hope-he-didnt-even-know-he-had/"
        },
        hoax: {
            title: "SHOCKING BOMBSHELL: Proof that Aliens Run Secret Deep State",
            text: "YOU WON'T BELIEVE THIS! Scientists discovered proof that secret aliens have taken over the mainstream media! They don't want you to know the shocking truth!"
        },
        cure: {
            title: "MIRACLE CURE: Secret Herb Destroys All Cancer in 48 Hours!",
            text: "Big pharma is hiding this unbelievable miracle cure! A secret wild plant cured thousands of terminal patients overnight, and corrupt doctors are trying to ban it!"
        }
    };

    // Tab Switching
    tabUrlBtn.addEventListener('click', () => {
        currentMode = 'url';
        tabUrlBtn.classList.add('active');
        tabTextBtn.classList.remove('active');
        tabUrlContent.classList.add('active');
        tabTextContent.classList.remove('active');
    });

    tabTextBtn.addEventListener('click', () => {
        currentMode = 'text';
        tabTextBtn.classList.add('active');
        tabUrlBtn.classList.remove('active');
        tabTextContent.classList.add('active');
        tabUrlContent.classList.remove('active');
    });

    // Preset Clicks
    document.querySelectorAll('.preset-chip').forEach(chip => {
        chip.addEventListener('click', () => {
            const key = chip.dataset.preset;
            const data = PRESETS[key];
            if (!data) return;

            if (data.url) {
                tabUrlBtn.click();
                inputUrl.value = data.url;
            } else {
                tabTextBtn.click();
                inputTitle.value = data.title || "";
                inputText.value = data.text || "";
            }
        });
    });

    // Verification Action
    btnVerify.addEventListener('click', async () => {
        let payload = {};

        if (currentMode === 'url') {
            const urlVal = inputUrl.value.trim();
            if (!urlVal) {
                alert("Please enter a valid article URL.");
                inputUrl.focus();
                return;
            }
            payload.url = urlVal;
        } else {
            const textVal = inputText.value.trim();
            if (!textVal) {
                alert("Please provide the news article text or claim to analyze.");
                inputText.focus();
                return;
            }
            payload.text = textVal;
            payload.title = inputTitle.value.trim();
        }

        // Set Loading State
        btnVerify.disabled = true;
        btnText.textContent = "Analyzing across 3 Pillars...";
        spinner.style.display = "block";

        try {
            const response = await fetch('/api/verify', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            if (!response.ok) {
                const errData = await response.json();
                throw new Error(errData.detail || "Verification request failed.");
            }

            const data = await response.json();
            renderResults(data);
        } catch (error) {
            alert(`Error: ${error.message}`);
        } finally {
            btnVerify.disabled = false;
            btnText.textContent = "Run Real-Time Verification";
            spinner.style.display = "none";
        }
    });

    function setText(id, text) {
        const el = document.getElementById(id);
        if (el) el.textContent = text !== undefined && text !== null ? text : "--";
    }

    function renderResults(data) {
        if (!data) return;
        resultsWrapper.style.display = 'block';

        // 1. Gauge Animation
        const score = data.credibility_score !== undefined ? data.credibility_score : 50;
        setText('gauge-value', score.toFixed(1));
        
        // 440 circumference (radius 70)
        const offset = 440 - (440 * (score / 100));
        if (gaugeProgress) {
            gaugeProgress.style.strokeDashoffset = offset;
        }

        // Color coding
        let verdictColor = '#10B981'; // Green
        let verdictClass = 'verdict-verified-true';
        let verdictText = 'Verified True';

        switch (data.verdict) {
            case 'VERIFIED_TRUE':
                verdictColor = '#10B981';
                verdictClass = 'verdict-verified-true';
                verdictText = 'Verified True';
                break;
            case 'LIKELY_REAL':
                verdictColor = '#0EA5E9';
                verdictClass = 'verdict-likely-real';
                verdictText = 'Likely Real';
                break;
            case 'UNVERIFIED_UNCERTAIN':
                verdictColor = '#F59E0B';
                verdictClass = 'verdict-unverified-uncertain';
                verdictText = 'Unverified / Uncertain';
                break;
            case 'LIKELY_FAKE':
                verdictColor = '#F97316';
                verdictClass = 'verdict-likely-fake';
                verdictText = 'Likely Fake / Sensational';
                break;
            case 'CONFIRMED_FAKE_OR_HOAX':
                verdictColor = '#EF4444';
                verdictClass = 'verdict-confirmed-fake-or-hoax';
                verdictText = 'Confirmed Fake / Hoax';
                break;
            case 'SATIRE_PARODY':
                verdictColor = '#A855F7';
                verdictClass = 'verdict-satire-parody';
                verdictText = 'Satire / Parody';
                break;
            default:
                verdictText = data.verdict || "Uncertain";
        }

        if (gaugeProgress) {
            gaugeProgress.style.stroke = verdictColor;
        }

        if (verdictBadge) {
            verdictBadge.className = `verdict-badge ${verdictClass}`;
            verdictBadge.textContent = verdictText;
        }
        setText('confidence-tag', `Confidence: ${data.confidence_level || 'MEDIUM'}`);
        setText('latency-tag', `⚡ Latency: ${data.execution_time_ms || 0} ms`);

        let headline = data.title;
        if (!headline && data.url) {
            try {
                headline = new URL(data.url).hostname;
            } catch (e) {
                headline = data.url;
            }
        }
        setText('article-headline', headline || "News Verification Report");
        setText('primary-reason', data.primary_reason || "Verification analysis complete.");

        // 2. Pillar 1: Domain
        const p1 = data.pillar_1_domain || {};
        setText('p1-domain', p1.domain || "N/A (Raw Text)");
        setText('p1-category', p1.status_label || "Unspecified");
        setText('p1-trust', `${((p1.trust_score !== undefined ? p1.trust_score : 0.5) * 100).toFixed(0)}%`);
        setText('p1-details', p1.details || "");

        // 3. Pillar 2: Stylometry
        const p2 = data.pillar_2_stylometry || {};
        const heuristics = p2.heuristics || {};
        const hasProb = p2.real_probability !== undefined && p2.fake_probability !== undefined;
        const realProb = hasProb ? (p2.real_probability * 100).toFixed(1) : "--";
        const fakeProb = hasProb ? (p2.fake_probability * 100).toFixed(1) : "--";
        
        setText('p2-style-label', p2.stylometric_label || (data.word_count === 0 ? "No body text to analyze" : "N/A"));
        setText('p2-real-prob', hasProb ? `${realProb}%` : "--");
        setText('p2-fake-prob', hasProb ? `${fakeProb}%` : "--");
        
        const barReal = document.getElementById('p2-bar-real');
        if (barReal) {
            barReal.style.width = hasProb ? `${realProb}%` : "0%";
            barReal.style.backgroundColor = verdictColor;
        }

        setText('p2-caps', heuristics.caps_ratio !== undefined ? `${(heuristics.caps_ratio * 100).toFixed(1)}%` : "--");
        setText('p2-excl', heuristics.exclamation_density !== undefined ? `${heuristics.exclamation_density} / 1k` : "--");

        // Sensational triggers
        const triggersContainer = document.getElementById('p2-triggers');
        if (triggersContainer) {
            triggersContainer.innerHTML = '';
            const hits = heuristics.sensational_hits || [];
            if (hits.length > 0) {
                hits.forEach(trigger => {
                    const span = document.createElement('span');
                    span.className = 'chip chip-danger';
                    span.textContent = trigger;
                    triggersContainer.appendChild(span);
                });
            } else {
                const span = document.createElement('span');
                span.className = 'chip chip-success';
                span.textContent = data.word_count === 0 ? 'No text extracted' : 'None (Journalistic tone)';
                triggersContainer.appendChild(span);
            }
        }

        // Keywords
        const keywordsContainer = document.getElementById('p2-keywords');
        if (keywordsContainer) {
            keywordsContainer.innerHTML = '';
            const topWords = p2.top_features || [];
            if (topWords.length > 0) {
                topWords.slice(0, 5).forEach(word => {
                    const span = document.createElement('span');
                    span.className = 'chip';
                    span.textContent = word;
                    keywordsContainer.appendChild(span);
                });
            } else {
                const span = document.createElement('span');
                span.className = 'chip';
                span.textContent = 'None';
                keywordsContainer.appendChild(span);
            }
        }

        // 4. Pillar 3: Fact Consensus
        const p3 = data.pillar_3_fact_consensus || {};
        const wires = p3.wire_consensus || {};
        const factChecks = p3.fact_checks || [];

        setText('p3-query', p3.query ? `"${p3.query}"` : "N/A");
        setText('p3-wire-count', `${wires.wire_count || 0} wire(s) corroborating`);

        const wiresListContainer = document.getElementById('p3-wire-badges');
        if (wiresListContainer) {
            wiresListContainer.innerHTML = '';
            const repSources = wires.reputable_sources || [];
            
            if (repSources.length > 0) {
                repSources.forEach(src => {
                    const span = document.createElement('span');
                    span.className = 'chip chip-success';
                    span.textContent = `✓ ${src}`;
                    wiresListContainer.appendChild(span);
                });
            } else {
                const wireBadges = [
                    { name: "Reuters", matched: wires.reuters_matched },
                    { name: "Associated Press", matched: wires.ap_matched },
                    { name: "BBC News", matched: wires.bbc_matched }
                ];
                wireBadges.forEach(wb => {
                    const span = document.createElement('span');
                    span.className = `chip ${wb.matched ? 'chip-success' : ''}`;
                    span.textContent = `${wb.matched ? '✓' : '—'} ${wb.name}`;
                    wiresListContainer.appendChild(span);
                });
            }
        }

        const debunksContainer = document.getElementById('p3-debunks-list');
        if (debunksContainer) {
            debunksContainer.innerHTML = '';
            if (factChecks.length > 0) {
                factChecks.slice(0, 2).forEach(fc => {
                    const div = document.createElement('div');
                    div.className = 'fact-item';
                    div.innerHTML = `
                        <div><span class="fact-publisher">${fc.publisher}</span>: <span class="fact-rating">${fc.rating}</span></div>
                        <div style="color: var(--text-secondary); margin-top: 2px;">${fc.claim}</div>
                    `;
                    debunksContainer.appendChild(div);
                });
            } else {
                debunksContainer.innerHTML = '<div style="color: var(--text-muted); font-size: 0.8rem; margin-top: 6px;">No recorded debunk records found for this query.</div>';
            }
        }

        // Smooth scroll to results
        resultsWrapper.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
});
