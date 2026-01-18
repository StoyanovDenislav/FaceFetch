document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("loginForm");
  const errorMessage = document.getElementById("errorMessage");
  const loading = document.getElementById("loading");
  const emailInput = document.getElementById("email");
  const rememberCheckbox = document.getElementById("rememberMe");

  const showError = (message) => {
    errorMessage.textContent = message;
    errorMessage.classList.add("show");
  };

  const hideError = () => {
    errorMessage.classList.remove("show");
  };

  const setLoading = (visible) => {
    loading.style.display = visible ? "block" : "none";
  };

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    hideError();
    setLoading(true);

    try {
      const email = emailInput.value;
      const password = document.getElementById("password").value;
      const rememberMe = rememberCheckbox.checked;

      const response = await fetch("/login", {
        method: "POST",
        headers: {
          "Content-Type": "application/x-www-form-urlencoded",
        },
        body: `email=${encodeURIComponent(email)}&password=${encodeURIComponent(password)}&rememberMe=${rememberMe}`,
      });

      if (response.ok) {
        if (rememberMe) {
          localStorage.setItem("rememberMe", "true");
          localStorage.setItem("email", email);
        } else {
          localStorage.removeItem("rememberMe");
          localStorage.removeItem("email");
        }
        window.location.href = "/";
      } else if (response.status === 401) {
        showError("Invalid email or password.");
      } else {
        const data = await response.json();
        showError(data.message || "Login failed. Please try again.");
      }
    } catch (error) {
      console.error("Error:", error);
      showError("An error occurred. Please try again later.");
    } finally {
      setLoading(false);
    }
  });

  document.querySelectorAll(".social-btn").forEach((button) => {
    button.addEventListener("click", (e) => {
      e.preventDefault();
      const provider = button.dataset.provider || "provider";
      const providerName =
        provider === "google" ? "Google" : provider === "github" ? "GitHub" : "This";
      alert(`${providerName} OAuth not configured yet. Please use email/password login.`);
    });
  });

  const remembered = localStorage.getItem("rememberMe");
  if (remembered) {
    emailInput.value = localStorage.getItem("email") || "";
    rememberCheckbox.checked = true;
  }
});
