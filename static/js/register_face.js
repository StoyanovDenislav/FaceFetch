document.addEventListener("DOMContentLoaded", () => {
  const video = document.getElementById("video");
  const canvas = document.getElementById("canvas");
  const captureBtn = document.getElementById("captureBtn");
  const registerBtn = document.getElementById("registerBtn");
  const userNameInput = document.getElementById("userName");
  const previewContainer = document.getElementById("previewContainer");
  const previewImage = document.getElementById("previewImage");
  const registerMessage = document.getElementById("registerMessage");
  const userMessage = document.getElementById("userMessage");
  const modeStatus = document.getElementById("modeStatus");
  const backToFeedBtn = document.getElementById("backToFeedBtn");
  const refreshBtn = document.getElementById("refreshBtn");
  const userList = document.getElementById("userList");

  let capturedImage = null;
  let stream = null;

  const showMessage = (elementId, message, type) => {
    const messageDiv = document.getElementById(elementId);
    messageDiv.innerHTML = `<div class="message ${type}">${message}</div>`;
    setTimeout(() => {
      messageDiv.innerHTML = "";
    }, 5000);
  };

  const stopStream = () => {
    if (!stream) return;
    stream.getTracks().forEach((track) => track.stop());
    stream = null;
  };

  const setModeStatus = (text, color) => {
    modeStatus.textContent = text;
    if (color) {
      modeStatus.style.color = color;
    }
  };

  const renderUsers = (users) => {
    if (!users || users.length === 0) {
      userList.innerHTML =
        '<p class="placeholder-text">No users registered yet</p>';
      return;
    }

    userList.innerHTML = "";

    users.forEach((user) => {
      const item = document.createElement("div");
      item.className = "user-item";

      const info = document.createElement("div");
      info.className = "user-info";

      const nameHeader = document.createElement("h3");
      nameHeader.innerHTML = `${user.name} <span class="status-badge ${
        user.active ? "active" : "inactive"
      }">${user.active ? "Active" : "Inactive"}</span>`;

      const meta = document.createElement("p");
      meta.textContent = `ID: ${user.id} | Face Profiles: ${user.face_profiles}`;

      info.appendChild(nameHeader);
      info.appendChild(meta);

      const deleteButton = document.createElement("button");
      deleteButton.className = "btn-danger user-delete";
      deleteButton.dataset.userId = user.id;
      deleteButton.dataset.userName = user.name;
      deleteButton.textContent = "Delete";

      deleteButton.addEventListener("click", () => handleDelete(user.id, user.name));

      item.appendChild(info);
      item.appendChild(deleteButton);

      userList.appendChild(item);
    });
  };

  async function initializeRegistrationMode() {
    try {
      setModeStatus("Initializing registration mode...", "");

      const enterResponse = await fetch("/api/control/enter_registration", {
        method: "POST",
      });
      const enterData = await enterResponse.json();

      if (!enterResponse.ok) {
        throw new Error(enterData.message || "Failed to enter registration mode");
      }

      console.log("Entered registration mode:", enterData.message);
      setModeStatus("Registration Mode - Camera Ready", "#ffc107");

      stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 640 },
          height: { ideal: 480 },
        },
      });
      video.srcObject = stream;

      console.log("Camera started successfully");
      captureBtn.disabled = false;
    } catch (err) {
      showMessage("registerMessage", "Error initializing: " + err.message, "error");
      setModeStatus("Error initializing registration mode", "#dc3545");
    }
  }

  captureBtn.addEventListener("click", () => {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0);

    capturedImage = canvas.toDataURL("image/jpeg", 0.95);
    previewImage.src = capturedImage;
    previewContainer.classList.remove("hidden");

    registerBtn.disabled = false;
    showMessage(
      "registerMessage",
      "Image captured! Enter your name and click Register.",
      "success",
    );
  });

  registerBtn.addEventListener("click", async () => {
    const name = userNameInput.value.trim();

    if (!name) {
      showMessage("registerMessage", "Please enter your name", "error");
      return;
    }

    if (!capturedImage) {
      showMessage("registerMessage", "Please capture an image first", "error");
      return;
    }

    registerBtn.disabled = true;
    registerBtn.textContent = "Registering...";

    try {
      const response = await fetch("/api/register", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          name,
          image: capturedImage,
        }),
      });

      const data = await response.json();

      if (response.ok) {
        showMessage(
          "registerMessage",
          `Success! ${name} registered successfully. Restarting face recognition...`,
          "success",
        );
        userNameInput.value = "";
        capturedImage = null;
        previewContainer.classList.add("hidden");
        registerBtn.disabled = true;

        stopStream();

        try {
          const exitResponse = await fetch("/api/control/exit_registration", {
            method: "POST",
          });
          const exitData = await exitResponse.json();

          if (exitResponse.ok) {
            console.log("Face recognition restarted:", exitData.message);
            setModeStatus("Ready - redirecting...", "#28a745");
            setTimeout(() => {
              window.location.href = "/";
            }, 1000);
          } else {
            console.error("Failed to restart module:", exitData.message);
            showMessage(
              "registerMessage",
              "Registration successful, but failed to restart module automatically.",
              "warning",
            );
          }
        } catch (exitErr) {
          console.error("Error exiting registration:", exitErr);
        }

        loadUsers();
      } else {
        showMessage("registerMessage", "Error: " + data.message, "error");
        registerBtn.disabled = false;
      }
    } catch (err) {
      showMessage("registerMessage", "Network error: " + err.message, "error");
      registerBtn.disabled = false;
    }

    registerBtn.textContent = "Register User";
  });

  async function loadUsers() {
    try {
      const response = await fetch("/api/users");
      const data = await response.json();

      if (data.users) {
        renderUsers(data.users);
      } else {
        userList.innerHTML =
          '<p class="placeholder-text">No users registered yet</p>';
      }
    } catch (err) {
      showMessage("userMessage", "Error loading users: " + err.message, "error");
    }
  }

  const handleDelete = async (userId, userName) => {
    if (!confirm(`Are you sure you want to delete ${userName}?`)) {
      return;
    }

    try {
      const response = await fetch(`/api/users/${userId}`, {
        method: "DELETE",
      });

      const data = await response.json();

      if (response.ok) {
        showMessage("userMessage", `${userName} deleted successfully`, "success");
        loadUsers();
      } else {
        showMessage("userMessage", "Error: " + data.message, "error");
      }
    } catch (err) {
      showMessage("userMessage", "Network error: " + err.message, "error");
    }
  };

  refreshBtn.addEventListener("click", loadUsers);

  backToFeedBtn.addEventListener("click", async (e) => {
    e.preventDefault();
    backToFeedBtn.disabled = true;
    backToFeedBtn.textContent = "Restoring Feed...";
    setModeStatus("Restoring face recognition...", "#ffc107");

    stopStream();

    try {
      const exitResponse = await fetch("/api/control/exit_registration", {
        method: "POST",
      });
      const exitData = await exitResponse.json();

      if (exitResponse.ok) {
        console.log("Face recognition restarted:", exitData.message);
        window.location.href = "/";
      } else {
        console.error("Failed to restart module:", exitData.message);
        showMessage(
          "registerMessage",
          "Failed to restart module automatically. Redirecting...",
          "warning",
        );
        setTimeout(() => (window.location.href = "/"), 2000);
      }
    } catch (err) {
      console.error("Error exiting registration:", err);
      window.location.href = "/";
    }
  });

  window.addEventListener("beforeunload", () => {
    stopStream();
  });

  initializeRegistrationMode();
  loadUsers();
});
