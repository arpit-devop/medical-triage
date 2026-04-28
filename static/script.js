document.addEventListener('DOMContentLoaded', () => {
    const inputArea = document.getElementById('symptoms-input');
    const analyzeBtn = document.getElementById('analyze-btn');
    const modelSelect = document.getElementById('model-select');
    const btnText = document.querySelector('.btn-text');
    const spinner = document.querySelector('.spinner');
    
    const resultsSection = document.getElementById('results-section');
    const specTarget = document.getElementById('predicted-specialty');
    const confFill = document.getElementById('confidence-fill');
    const confText = document.getElementById('confidence-text');
    const timeBadge = document.getElementById('response-time');
    const urgentAlert = document.getElementById('urgent-alert');
    const statModel = document.getElementById('stat-model');
    const keywordsContainer = document.getElementById('keywords-container');
    const iconContainer = document.getElementById('specialty-icon');

    // Medical icons map matching specialties
    const icons = {
        'Cardiology': `<svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z"></path></svg>`,
        'Neurology': `<svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><path d="M12 2a14.5 14.5 0 0 0 0 20 14.5 14.5 0 0 0 0-20"></path><path d="M2 12h20"></path></svg>`,
        'Orthopedics': `<svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="12" y1="5" x2="12" y2="19"></line><line x1="5" y1="12" x2="19" y2="12"></line></svg>`,
        'Gastroenterology': `<svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20.59 13.41l-7.17 7.17a2 2 0 0 1-2.83 0L2 12V2h10l8.59 8.59a2 2 0 0 1 0 2.82z"></path><line x1="7" y1="7" x2="7.01" y2="7"></line></svg>`,
        'Pulmonology': `<svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M9.59 4.59A2 2 0 1 1 11 8H2m10.59 11.41A2 2 0 1 0 14 16H2m15.73-8.27A2.5 2.5 0 1 1 19.5 12H2"></path></svg>`,
        'General Practice': `<svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline><line x1="12" y1="18" x2="12" y2="12"></line><line x1="9" y1="15" x2="15" y2="15"></line></svg>`
    };

    const defaultIcon = `<svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 12h-4l-3 9L9 3l-3 9H2"></path></svg>`;

    analyzeBtn.addEventListener('click', async () => {
        const text = inputArea.value.trim();
        if (!text) {
            inputArea.style.borderColor = 'var(--danger)';
            setTimeout(() => inputArea.style.borderColor = 'var(--panel-border)', 1000);
            return;
        }

        // Set Loading State
        analyzeBtn.disabled = true;
        btnText.style.display = 'none';
        spinner.style.display = 'block';
        
        // Hide results temporarily for transiton
        resultsSection.classList.remove('visible');
        confFill.style.width = '0%';

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ 
                    symptoms: text,
                    model_type: modelSelect.value
                })
            });

            const data = await response.json();

            if (response.ok) {
                // Populate UI
                specTarget.innerText = data.specialty;
                timeBadge.innerText = `${data.response_time_ms} ms`;
                confText.innerText = `${data.confidence}% Confidence`;
                statModel.innerText = data.model_used;
                
                // Icon
                iconContainer.innerHTML = icons[data.specialty] || defaultIcon;

                // Keywords
                keywordsContainer.innerHTML = '';
                if (data.keywords && data.keywords.length > 0) {
                    data.keywords.forEach(kw => {
                        const span = document.createElement('span');
                        span.className = 'keyword-tag';
                        span.innerText = kw;
                        keywordsContainer.appendChild(span);
                    });
                } else {
                    keywordsContainer.innerText = 'None detected';
                }

                // Urgent Flag
                if (data.urgent) {
                    urgentAlert.classList.remove('hidden');
                } else {
                    urgentAlert.classList.add('hidden');
                }

                // Reveal Results
                resultsSection.classList.add('visible');
                
                // Animate confidence bar
                setTimeout(() => {
                    confFill.style.width = `${data.confidence}%`;
                }, 300);

            } else {
                alert(`Error: ${data.error || 'Failed to process request.'}`);
            }

        } catch (err) {
            console.error(err);
            alert("Network error. Could not connect to API.");
        } finally {
            // Restore button
            analyzeBtn.disabled = false;
            btnText.style.display = 'inline';
            spinner.style.display = 'none';
        }
    });

    // Handle enter key
    inputArea.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
            analyzeBtn.click();
        }
    });
});
