let audioContext;
let oscillator1;
let oscillator2;
let oscillator3;
let gainNode;
let isAlarmPlaying = false;

let alarmSound;
let alarmBanner;
let audioNotice;

let alarmActive = false;
let lastAlertCount = 0;
let audioEnabled = false;
let unknownDetectionStart = null;
let lastAlarmTime = 0;

const ALARM_COOLDOWN = 30000;
const DETECTION_DELAY = 2500;

function createEASAlarm() {
  if (!audioContext) {
    audioContext = new (window.AudioContext || window.webkitAudioContext)();
  }

  stopEASAlarm();

  oscillator1 = audioContext.createOscillator();
  oscillator2 = audioContext.createOscillator();
  oscillator3 = audioContext.createOscillator();
  gainNode = audioContext.createGain();

  oscillator1.frequency.value = 853;
  oscillator2.frequency.value = 960;
  oscillator3.frequency.value = 1050;

  oscillator1.type = "square";
  oscillator2.type = "square";
  oscillator3.type = "square";

  gainNode.gain.value = 0.3;

  oscillator1.connect(gainNode);
  oscillator2.connect(gainNode);
  oscillator3.connect(gainNode);
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
  if (oscillator1) {
    try {
      oscillator1.stop();
    } catch (e) {
      /* no-op */
    }
    oscillator1 = null;
  }
  if (oscillator2) {
    try {
      oscillator2.stop();
    } catch (e) {
      /* no-op */
    }
    oscillator2 = null;
  }
  if (oscillator3) {
    try {
      oscillator3.stop();
    } catch (e) {
      /* no-op */
    }
    oscillator3 = null;
  }
  isAlarmPlaying = false;
}

function stopAlarm() {
  alarmActive = false;
  if (alarmBanner) {
    alarmBanner.style.display = "none";
  }
  stopEASAlarm();
  if (alarmSound) {
    alarmSound.pause();
    alarmSound.currentTime = 0;
  }
}

function triggerAlarm() {
  const currentTime = Date.now();
  if (currentTime - lastAlarmTime < ALARM_COOLDOWN) {
    return;
  }

  if (!alarmActive) {
    alarmActive = true;
    lastAlarmTime = currentTime;
    if (alarmBanner) {
      alarmBanner.style.display = "block";
    }

    if (audioEnabled) {
      createEASAlarm();
    } else {
      console.log("Audio not enabled yet - click anywhere to enable");
    }

    setTimeout(() => {
      stopAlarm();
    }, 15000);
  }
}

const dismissAlarm = () => {
  stopAlarm();
};

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

function buildAlertItem(alert, index) {
  const alertItem = document.createElement("div");
  alertItem.className = `alert-item ${alert.type}`;

  const header = document.createElement("div");
  header.className = "alert-header";

  const typeSpan = document.createElement("span");
  typeSpan.className = "alert-type";
  typeSpan.textContent =
    alert.type === "spoof" ? "dYs® SPOOF DETECTED" : "ƒsÿ‹,? UNKNOWN PERSON";

  const date = new Date(alert.timestamp * 1000);
  const timeString = date.toLocaleTimeString();

  const timeSpan = document.createElement("span");
  timeSpan.className = "alert-time";
  timeSpan.textContent = timeString;

  header.appendChild(typeSpan);
  header.appendChild(timeSpan);

  const message = document.createElement("div");
  message.className = "alert-message";
  message.textContent = alert.message;

  const dismiss = document.createElement("button");
  dismiss.className = "alert-dismiss";
  dismiss.textContent = "Dismiss";
  dismiss.addEventListener("click", () => acknowledgeAlert(index));

  alertItem.appendChild(header);
  alertItem.appendChild(message);
  alertItem.appendChild(dismiss);

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
      if (alarmActive && lastAlertCount > 0) {
        stopAlarm();
      }
    } else {
      if (data.total_alerts > lastAlertCount) {
        if (unknownDetectionStart === null) {
          unknownDetectionStart = currentTime;
        }

        const detectionDuration = currentTime - unknownDetectionStart;
        if (detectionDuration >= DETECTION_DELAY) {
          triggerAlarm();
          unknownDetectionStart = null;
        } else {
          console.log(
            "Detection duration:",
            detectionDuration,
            "ms (need",
            DETECTION_DELAY,
            "ms)",
          );
        }
      }
      lastAlertCount = data.total_alerts;

      container.innerHTML = "";

      data.alerts.forEach((alert, index) => {
        const alertNode = buildAlertItem(alert, index);
        container.appendChild(alertNode);
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

    const now = new Date(data.timestamp * 1000);
    document.getElementById("lastUpdate").textContent = now.toLocaleTimeString();

    const container = document.getElementById("facesContainer");

    if (data.total_faces === 0) {
      container.innerHTML = '<div class="no-faces">No faces detected</div>';
    } else {
      container.innerHTML = "";

      data.faces.forEach((face, index) => {
        const faceCard = document.createElement("div");
        faceCard.className = `face-card ${face.state}`;

        const badgeClass =
          face.state === "known"
            ? "badge-known"
            : face.state === "unknown"
              ? "badge-unknown"
              : face.state === "spoof"
                ? "badge-spoof"
                : "badge-pending";

        const liveClass = face.is_live ? "live" : "not-live";
        const liveText = face.is_live ? "Live" : "Not Live";

        faceCard.innerHTML = `
              <div class="face-header">
                  <div class="face-name">Face #${index + 1}</div>
                  <span class="face-badge ${badgeClass}">${face.state}</span>
              </div>
              <div class="face-details">
                  <strong>Name:</strong> ${face.name || "Unknown"}<br>
                  ${face.confidence ? `<strong>Confidence:</strong> ${face.confidence}<br>` : ""}
                  <strong>Status:</strong> <span class="live-indicator ${liveClass}"></span>${liveText}<br>
                  <strong>Location:</strong> Top: ${face.location.top}, Left: ${face.location.left}
              </div>
          `;

        container.appendChild(faceCard);
      });
    }

    const connectionStatus = document.getElementById("connectionStatus");
    connectionStatus.textContent = "Connected";
    connectionStatus.style.color = "#28a745";

    const jsonDataEl = document.getElementById("jsonData");
    if (jsonDataEl) {
      jsonDataEl.textContent = JSON.stringify(data, null, 2);
    }
  } catch (error) {
    console.error("Error fetching detections:", error);
    const connectionStatus = document.getElementById("connectionStatus");
    connectionStatus.textContent = "Disconnected";
    connectionStatus.style.color = "#dc3545";
  }
}

async function updateSystemStatus() {
  try {
    const response = await fetch("/api/status");
    const data = await response.json();

    document.getElementById("cameraType").textContent = data.camera_type;
    document.getElementById("knownFaces").textContent =
      data.known_faces + " (" + data.faces_loaded.join(", ") + ")";
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

      data.history.slice(0, 20).forEach((entry) => {
        const historyItem = document.createElement("div");
        historyItem.className = `history-item ${entry.state}`;

        const date = new Date(entry.timestamp * 1000);
        const timeString = date.toLocaleTimeString();

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

async function clearHistory() {
  try {
    await fetch("/api/history/clear", { method: "POST" });
    updateHistory();
  } catch (error) {
    console.error("Error clearing history:", error);
  }
}

async function ensureFaceRecognitionRunning() {
  try {
    const response = await fetch("/api/status");
    const data = await response.json();

    if (response.status === 500 || !data.faces_loaded) {
      console.log(
        "Face recognition may not be running, attempting to restart...",
      );
      const exitResponse = await fetch("/api/control/exit_registration", {
        method: "POST",
      });
      const exitData = await exitResponse.json();

      if (exitResponse.ok) {
        console.log("バ. Face recognition restarted:", exitData.message);
        console.log("Faces loaded:", exitData.faces);
      } else {
        console.error("ƒ?O Failed to restart face recognition:", exitData.message);
      }
    }
  } catch (error) {
    console.error("Error checking face recognition status:", error);
  }
}

function enableAudio() {
  audioEnabled = true;
  if (audioNotice) {
    audioNotice.style.display = "none";
  }
  document.removeEventListener("click", enableAudio);
  console.log("Audio enabled");
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
    if (!audioEnabled && audioNotice) {
      audioNotice.style.display = "block";
    }
  }, 2000);

  if (clearAlertsBtn) {
    clearAlertsBtn.addEventListener("click", clearAllAlerts);
  }

  if (clearHistoryBtn) {
    clearHistoryBtn.addEventListener("click", clearHistory);
  }

  if (dismissAlarmBtn) {
    dismissAlarmBtn.addEventListener("click", dismissAlarm);
  }

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
