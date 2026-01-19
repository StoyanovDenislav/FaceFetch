let audioContext, oscillator1, oscillator2, oscillator3, gainNode;
let isAlarmPlaying = false;
let alarmSound, alarmBanner, audioNotice;
let alarmActive = false;
let lastAlertCount = 0;
let audioEnabled = false;
let unknownDetectionStart = null;
let lastAlarmTime = 0;

const ALARM_COOLDOWN = 30000;
const DETECTION_DELAY = 2500;
const EAS_FREQUENCIES = [853, 960, 1050];

function createEASAlarm() {
  if (!audioContext) {
    audioContext = new (window.AudioContext || window.webkitAudioContext)();
  }

  stopEASAlarm();

  oscillator1 = audioContext.createOscillator();
  oscillator2 = audioContext.createOscillator();
  oscillator3 = audioContext.createOscillator();
  gainNode = audioContext.createGain();

  oscillator1.frequency.value = EAS_FREQUENCIES[0];
  oscillator2.frequency.value = EAS_FREQUENCIES[1];
  oscillator3.frequency.value = EAS_FREQUENCIES[2];

  [oscillator1, oscillator2, oscillator3].forEach(osc => {
    osc.type = "square";
    osc.connect(gainNode);
  });

  gainNode.gain.value = 0.3;
  gainNode.connect(audioContext.destination);

  oscillator1.start();
  oscillator2.start();
  oscillator3.start();

  const startTime = audioContext.currentTime;
  for (let i = 0; i < 100; i++) {
    gainNode.gain.setValueAtTime(0.3, startTime + i * 0.5);
    gainNode.gain.setValueAtTime(0.0, startTime + i * 0.5 + 0.25);
  }

  isAlarmPlaying = true;
}

function stopEASAlarm() {
  [oscillator1, oscillator2, oscillator3].forEach((osc, i) => {
    if (osc) {
      try { osc.stop(); } catch (e) { }
      if (i === 0) oscillator1 = null;
      if (i === 1) oscillator2 = null;
      if (i === 2) oscillator3 = null;
    }
  });
  isAlarmPlaying = false;
}

function stopAlarm() {
  alarmActive = false;
  if (alarmBanner) alarmBanner.style.display = "none";
  stopEASAlarm();
  if (alarmSound) {
    alarmSound.pause();
    alarmSound.currentTime = 0;
  }
}

function triggerAlarm() {
  const currentTime = Date.now();
  if (currentTime - lastAlarmTime < ALARM_COOLDOWN) return;

  if (!alarmActive) {
    alarmActive = true;
    lastAlarmTime = currentTime;
    if (alarmBanner) alarmBanner.style.display = "block";

    if (audioEnabled) {
      createEASAlarm();
    }

    setTimeout(stopAlarm, 15000);
  }
}

function dismissAlarm() {
  stopAlarm();
}

async function acknowledgeAlert(index) {
  try {
    await fetch(`/api/alerts/${index}/acknowledge`, { method: "POST" });
    updateAlerts();
  } catch (error) {
    console.error("Error acknowledging alert:", error);
  }
}

async function clearAllAlerts() {
  try {
    await fetch("/api/alerts/clear", { method: "POST" });
    stopAlarm();
    lastAlertCount = 0;
    updateAlerts();
  } catch (error) {
    console.error("Error clearing alerts:", error);
  }
}

async function clearHistory() {
  try {
    await fetch("/api/history/clear", { method: "POST" });
    updateHistory();
  } catch (error) {
    console.error("Error clearing history:", error);
  }
}

function buildAlertItem(alert, index) {
  const alertItem = document.createElement("div");
  alertItem.className = `alert-item ${alert.type}`;

  const header = document.createElement("div");
  header.className = "alert-header";

  const typeSpan = document.createElement("span");
  typeSpan.className = "alert-type";
  typeSpan.textContent = alert.type === "spoof" ? "dYs® SPOOF DETECTED" : "ƒsÿ‹,? UNKNOWN PERSON";

  const timeSpan = document.createElement("span");
  timeSpan.className = "alert-time";
  timeSpan.textContent = new Date(alert.timestamp * 1000).toLocaleTimeString();

  header.append(typeSpan, timeSpan);

  const message = document.createElement("div");
  message.className = "alert-message";
  message.textContent = alert.message;

  const dismiss = document.createElement("button");
  dismiss.className = "alert-dismiss";
  dismiss.textContent = "Dismiss";
  dismiss.onclick = () => acknowledgeAlert(index);

  alertItem.append(header, message, dismiss);
  return alertItem;
}

async function updateAlerts() {
  try {
    const response = await fetch("/api/alerts?unacknowledged=true");
    const data = await response.json();
    const container = document.getElementById("alertsContainer");
    const currentTime = Date.now();

    if (data.total_alerts === 0) {
      container.innerHTML = '<div class="no-faces">No alerts</div>';
      unknownDetectionStart = null;
      if (alarmActive && lastAlertCount > 0) stopAlarm();
    } else {
      if (data.total_alerts > lastAlertCount) {
        if (unknownDetectionStart === null) {
          unknownDetectionStart = currentTime;
        }

        const detectionDuration = currentTime - unknownDetectionStart;
        if (detectionDuration >= DETECTION_DELAY) {
          triggerAlarm();
          unknownDetectionStart = null;
        }
      }

      lastAlertCount = data.total_alerts;
      container.innerHTML = "";
      data.alerts.forEach((alert, index) => {
        container.appendChild(buildAlertItem(alert, index));
      });
    }
  } catch (error) {
    console.error("Error fetching alerts:", error);
  }
}

async function updateDetections() {
  try {
    const response = await fetch("/api/detections");
    const data = await response.json();

    document.getElementById("totalFaces").textContent = data.total_faces;
    document.getElementById("lastUpdate").textContent = new Date(data.timestamp * 1000).toLocaleTimeString();

    const container = document.getElementById("facesContainer");

    if (data.total_faces === 0) {
      container.innerHTML = '<div class="no-faces">No faces detected</div>';
    } else {
      container.innerHTML = "";
      data.faces.forEach((face, index) => {
        const faceCard = document.createElement("div");
        faceCard.className = `face-card ${face.state}`;

        const badgeMap = {
          known: "badge-known",
          unknown: "badge-unknown",
          spoof: "badge-spoof",
          pending: "badge-pending"
        };
        const badgeClass = badgeMap[face.state] || "badge-pending";
        const liveClass = face.is_live ? "live" : "not-live";
        const liveText = face.is_live ? "Live" : "Not Live";

        faceCard.innerHTML = `
          <div class="face-header">
            <div class="face-name">Face #${index + 1}</div>
            <span class="face-badge ${badgeClass}">${face.state}</span>
          </div>
          <div class="face-details">
            <strong>Name:</strong> ${face.name || "Unknown"}

            ${face.confidence ? `<strong>Confidence:</strong> ${face.confidence}
` : ""}
            <strong>Status:</strong> <span class="live-indicator ${liveClass}"></span>${liveText}

            <strong>Location:</strong> Top: ${face.location.top}, Left: ${face.location.left}
          </div>
        `;

        container.appendChild(faceCard);
      });
    }

    updateConnectionStatus(true);

    const jsonDataEl = document.getElementById("jsonData");
    if (jsonDataEl) jsonDataEl.textContent = JSON.stringify(data, null, 2);
  } catch (error) {
    console.error("Error fetching detections:", error);
    updateConnectionStatus(false);
  }
}

async function updateSystemStatus() {
  try {
    const response = await fetch("/api/status");
    const data = await response.json();
    document.getElementById("cameraType").textContent = data.camera_type;
    document.getElementById("knownFaces").textContent = `${data.known_faces} (${data.faces_loaded.join(", ")})`;
  } catch (error) {
    console.error("Error fetching system status:", error);
  }
}

async function updateHistory() {
  try {
    const response = await fetch("/api/history");
    const data = await response.json();
    const container = document.getElementById("historyContainer");

    if (data.total_entries === 0) {
      container.innerHTML = '<div class="no-faces">No history yet</div>';
    } else {
      container.innerHTML = "";
      data.history.slice(0, 20).forEach(entry => {
        const historyItem = document.createElement("div");
        historyItem.className = `history-item ${entry.state}`;
        const timeString = new Date(entry.timestamp * 1000).toLocaleTimeString();

        historyItem.innerHTML = `
          <div>
            <div class="history-time">${timeString}</div>
            <div class="history-name">${entry.name || "Unknown"}</div>
          </div>
          <div>
            <span class="face-badge badge-${entry.state}">${entry.state}</span>
          </div>
        `;

        container.appendChild(historyItem);
      });
    }
  } catch (error) {
    console.error("Error fetching history:", error);
  }
}

function updateConnectionStatus(isConnected) {
  const status = document.getElementById("connectionStatus");
  status.textContent = isConnected ? "Connected" : "Disconnected";
  status.style.color = isConnected ? "#57c96d" : "#ff6473";
}

async function ensureFaceRecognitionRunning() {
  try {
    const response = await fetch("/api/status");
    const data = await response.json();

    if (response.status === 500 || !data.faces_loaded) {
      const exitResponse = await fetch("/api/control/exit_registration", { method: "POST" });
      const exitData = await exitResponse.json();
      if (exitResponse.ok) {
        console.log("Face recognition restarted:", exitData.message);
      }
    }
  } catch (error) {
    console.error("Error checking face recognition status:", error);
  }
}

function enableAudio() {
  audioEnabled = true;
  if (audioNotice) audioNotice.style.display = "none";
  document.removeEventListener("click", enableAudio);
}

document.addEventListener("DOMContentLoaded", () => {
  alarmSound = document.getElementById("alarmSound");
  alarmBanner = document.getElementById("alarmBanner");
  audioNotice = document.getElementById("audioNotice");

  const clearAlertsBtn = document.getElementById("clearAlerts");
  const clearHistoryBtn = document.getElementById("clearHistory");
  const dismissAlarmBtn = document.getElementById("dismissAlarmBtn");

  document.addEventListener("click", enableAudio, { once: true });

  setTimeout(() => {
    if (!audioEnabled && audioNotice) audioNotice.style.display = "block";
  }, 2000);

  if (clearAlertsBtn) clearAlertsBtn.onclick = clearAllAlerts;
  if (clearHistoryBtn) clearHistoryBtn.onclick = clearHistory;
  if (dismissAlarmBtn) dismissAlarmBtn.onclick = dismissAlarm;

  ensureFaceRecognitionRunning().then(() => {
    updateDetections();
    updateSystemStatus();
    updateHistory();
    updateAlerts();
  });

  setInterval(updateDetections, 1000);
  setInterval(updateAlerts, 1000);
  setInterval(updateHistory, 2000);
  setInterval(updateSystemStatus, 10000);
});