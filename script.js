/**
 * AI VISION: HAND TRACKING & EMOTION DETECTION SYSTEM
 * Interactive Frontend JavaScript Controller
 */

document.addEventListener('DOMContentLoaded', () => {
    // UI Elements
    const btnStartCamera = document.getElementById('btnStartCamera');
    const btnStopCamera = document.getElementById('btnStopCamera');
    const btnPlaceholderStart = document.getElementById('btnPlaceholderStart');
    const btnToggleFullscreen = document.getElementById('btnToggleFullscreen');

    const videoFeed = document.getElementById('videoFeed');
    const videoPlaceholder = document.getElementById('videoPlaceholder');
    const streamHudOverlay = document.getElementById('streamHudOverlay');
    const systemStatusBadge = document.getElementById('systemStatusBadge');
    const statusText = document.getElementById('statusText');

    // Telemetry Elements
    const fpsDisplay = document.getElementById('fpsDisplay');
    const primaryEmotionEmoji = document.getElementById('primaryEmotionEmoji');
    const primaryEmotionName = document.getElementById('primaryEmotionName');
    const primaryEmotionConf = document.getElementById('primaryEmotionConf');
    const emotionProgressBar = document.getElementById('emotionProgressBar');
    const emotionMeterVal = document.getElementById('emotionMeterVal');
    const emotionSpectrumList = document.getElementById('emotionSpectrumList');

    const handsCountBadge = document.getElementById('handsCountBadge');
    const totalFingersCount = document.getElementById('totalFingersCount');
    const detectedHandsCount = document.getElementById('detectedHandsCount');
    const handsListContainer = document.getElementById('handsListContainer');

    // State Variables
    let pollInterval = null;
    let isCameraActive = false;

    // Toast Notification helper
    const showToast = (title, message, isError = false) => {
        const toastEl = document.getElementById('cyberToast');
        const toastTitle = document.getElementById('toastTitle');
        const toastBody = document.getElementById('toastBody');

        toastTitle.textContent = title;
        toastBody.textContent = message;

        if (isError) {
            toastEl.style.borderColor = '#ff3366';
        } else {
            toastEl.style.borderColor = '#00f0ff';
        }

        const toast = new bootstrap.Toast(toastEl);
        toast.show();
    };

    // 1. Start Camera Handler
    const startCamera = async () => {
        btnStartCamera.disabled = true;
        btnPlaceholderStart.disabled = true;

        try {
            const response = await fetch('/api/start_camera', { method: 'POST' });
            const data = await response.json();

            if (data.success) {
                isCameraActive = true;

                // Update Video Stream
                videoFeed.src = `/video_feed?t=${new Date().getTime()}`;
                videoFeed.classList.remove('d-none');
                videoPlaceholder.classList.add('d-none');
                streamHudOverlay.classList.remove('d-none');

                // Update Buttons & Status Badge
                btnStartCamera.classList.add('d-none');
                btnStopCamera.classList.remove('d-none');
                
                systemStatusBadge.className = 'status-pill online';
                statusText.textContent = 'AI STREAM ACTIVE';

                showToast('Camera Activated', 'Real-time hand tracking and emotion detection engine is running.');

                // Start Telemetry Polling Loop
                startPolling();
            } else {
                showToast('Camera Error', data.message || 'Failed to start camera.', true);
            }
        } catch (error) {
            console.error('Error starting camera:', error);
            showToast('Connection Error', 'Could not reach server endpoint.', true);
        } finally {
            btnStartCamera.disabled = false;
            btnPlaceholderStart.disabled = false;
        }
    };

    // 2. Stop Camera Handler
    const stopCamera = async () => {
        btnStopCamera.disabled = true;

        try {
            const response = await fetch('/api/stop_camera', { method: 'POST' });
            const data = await response.json();

            if (data.success) {
                isCameraActive = false;

                // Stop Stream and reset view
                videoFeed.src = '';
                videoFeed.classList.add('d-none');
                videoPlaceholder.classList.remove('d-none');
                streamHudOverlay.classList.add('d-none');

                // Update Controls
                btnStartCamera.classList.remove('d-none');
                btnStopCamera.classList.add('d-none');

                systemStatusBadge.className = 'status-pill offline';
                statusText.textContent = 'SYSTEM READY';

                stopPolling();
                resetTelemetryDisplays();

                showToast('Camera Stopped', 'Webcam hardware released cleanly.');
            }
        } catch (error) {
            console.error('Error stopping camera:', error);
        } finally {
            btnStopCamera.disabled = false;
        }
    };

    // 3. Telemetry Polling Loop (~10 Hz)
    const startPolling = () => {
        if (pollInterval) clearInterval(pollInterval);
        pollInterval = setInterval(fetchTelemetry, 100);
    };

    const stopPolling = () => {
        if (pollInterval) {
            clearInterval(pollInterval);
            pollInterval = null;
        }
    };

    const fetchTelemetry = async () => {
        if (!isCameraActive) return;

        try {
            const response = await fetch('/api/data');
            const data = await response.json();

            updateDashboardUI(data);
        } catch (err) {
            console.error('Error fetching telemetry:', err);
        }
    };

    // 4. UI Dashboard Renderer
    const updateDashboardUI = (data) => {
        // Update FPS
        if (fpsDisplay) {
            fpsDisplay.innerHTML = `<i class="fa-solid fa-gauge-high me-1 text-cyber-emerald"></i> FPS: ${data.fps || '0.0'}`;
        }

        // --- Emotion Analysis Update ---
        const emotion = data.emotion || {};
        if (emotion.detected) {
            primaryEmotionEmoji.textContent = emotion.emoji || '😐';
            primaryEmotionName.textContent = emotion.primary || 'Neutral';
            primaryEmotionConf.textContent = `${emotion.confidence}%`;
            
            emotionProgressBar.style.width = `${emotion.confidence}%`;
            emotionMeterVal.textContent = `${emotion.confidence}%`;

            renderEmotionSpectrum(emotion.all_emotions);
        } else {
            primaryEmotionEmoji.textContent = '🔍';
            primaryEmotionName.textContent = 'No Face';
            primaryEmotionConf.textContent = '0%';
            emotionProgressBar.style.width = '0%';
            emotionMeterVal.textContent = '0%';

            renderEmotionSpectrum({});
        }

        // --- Hand Tracking & Gesture Update ---
        const handsCount = data.hands_detected || 0;
        const fingersCount = data.total_fingers || 0;

        handsCountBadge.textContent = `${handsCount} Hand${handsCount !== 1 ? 's' : ''}`;
        detectedHandsCount.textContent = handsCount;
        totalFingersCount.textContent = fingersCount;

        renderHandsList(data.hands || []);
    };

    // Render Emotion Spectrum Bar Graph
    const renderEmotionSpectrum = (emotionsDict) => {
        if (!emotionsDict || Object.keys(emotionsDict).length === 0) {
            emotionSpectrumList.innerHTML = `
                <div class="text-center py-3 text-cyber-muted small">
                    <i class="fa-solid fa-user-slash mb-1 d-block opacity-50"></i> Position face in frame...
                </div>
            `;
            return;
        }

        const sortedEmotions = Object.entries(emotionsDict).sort((a, b) => b[1] - a[1]);

        const emotionColors = {
            Happy: '#00ff88',
            Surprise: '#00f0ff',
            Neutral: '#94a3b8',
            Sad: '#3b82f6',
            Angry: '#ff3366',
            Fear: '#a855f7',
            Disgust: '#eab308'
        };

        let html = '';
        sortedEmotions.forEach(([emo, pct]) => {
            const color = emotionColors[emo] || '#00f0ff';
            html += `
                <div class="spectrum-item">
                    <div class="d-flex justify-content-between spectrum-label">
                        <span>${emo}</span>
                        <span>${pct}%</span>
                    </div>
                    <div class="spectrum-progress">
                        <div class="spectrum-bar" style="width: ${pct}%; background-color: ${color}; box-shadow: 0 0 6px ${color};"></div>
                    </div>
                </div>
            `;
        });

        emotionSpectrumList.innerHTML = html;
    };

    // Render Hands and Gestures List
    const renderHandsList = (handsList) => {
        if (!handsList || handsList.length === 0) {
            handsListContainer.innerHTML = `
                <div class="no-hands-placeholder text-center py-4 text-cyber-muted">
                    <i class="fa-solid fa-hand-sparkles display-6 mb-2 opacity-50"></i>
                    <p class="mb-0 small">No hands detected in webcam view</p>
                </div>
            `;
            return;
        }

        let html = '';
        handsList.forEach((hand) => {
            const sideColor = hand.handedness === 'Right' ? 'text-cyber-cyan' : 'text-cyber-magenta';
            html += `
                <div class="hand-badge-card d-flex align-items-center justify-content-between">
                    <div class="d-flex align-items-center gap-3">
                        <div class="gesture-emoji-tag">${hand.emoji || '🖐️'}</div>
                        <div>
                            <div class="font-orbitron text-white small">${hand.gesture || 'Hand Active'}</div>
                            <small class="${sideColor} font-orbitron me-2">${hand.handedness} Hand</small>
                            <small class="text-cyber-muted">${hand.raised_count} Raised</small>
                        </div>
                    </div>
                    <div class="text-end">
                        <span class="badge bg-cyber-badge">${hand.confidence}%</span>
                    </div>
                </div>
            `;
        });

        handsListContainer.innerHTML = html;
    };

    // Reset UI to clean initial state
    const resetTelemetryDisplays = () => {
        if (fpsDisplay) fpsDisplay.innerHTML = `<i class="fa-solid fa-gauge-high me-1 text-cyber-emerald"></i> FPS: 0.0`;
        primaryEmotionEmoji.textContent = '😐';
        primaryEmotionName.textContent = 'Neutral';
        primaryEmotionConf.textContent = '0%';
        emotionProgressBar.style.width = '0%';
        emotionMeterVal.textContent = '0%';

        emotionSpectrumList.innerHTML = `<div class="text-center py-3 text-cyber-muted small">Camera Offline</div>`;
        handsCountBadge.textContent = '0 Hands';
        detectedHandsCount.textContent = '0';
        totalFingersCount.textContent = '0';
        
        handsListContainer.innerHTML = `
            <div class="no-hands-placeholder text-center py-4 text-cyber-muted">
                <i class="fa-solid fa-hand-sparkles display-6 mb-2 opacity-50"></i>
                <p class="mb-0 small">No hands detected in webcam view</p>
            </div>
        `;
    };

    // 5. Event Listeners
    btnStartCamera.addEventListener('click', startCamera);
    btnPlaceholderStart.addEventListener('click', startCamera);
    btnStopCamera.addEventListener('click', stopCamera);

    btnToggleFullscreen.addEventListener('click', () => {
        if (!document.fullscreenElement) {
            videoFeed.requestFullscreen().catch(err => {
                showToast('Fullscreen Error', 'Could not open video stream in fullscreen mode.');
            });
        } else {
            document.exitFullscreen();
        }
    });
});
