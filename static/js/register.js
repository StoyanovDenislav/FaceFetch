document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("registerForm");
  const errorMessage = document.getElementById("errorMessage");
  const successMessage = document.getElementById("successMessage");
  const loading = document.getElementById("loading");
  const passwordInput = document.getElementById("password");
  const strengthIndicator = document.getElementById("strengthIndicator");
  const strengthText = document.getElementById("strengthText");
  const registerBtn = document.getElementById("registerBtn");

  const showError = (message) => {
    errorMessage.textContent = message;
    errorMessage.classList.add("show");
  };

  const showSuccess = (message) => {
    successMessage.textContent = message;
    successMessage.classList.add("show");
  };

  const clearMessages = () => {
    errorMessage.classList.remove("show");
    successMessage.classList.remove("show");
  };

  const toggleLoading = (visible) => {
    loading.style.display = visible ? "block" : "none";
  };

  const updateStrength = () => {
    const password = passwordInput.value;
    let strength = 0;
    let strengthLabel = "Weak";

    if (password.length >= 8) strength++;
    if (/[a-z]/.test(password) && /[A-Z]/.test(password)) strength++;
    if (/\d/.test(password)) strength++;
    if (/[^a-zA-Z\d]/.test(password)) strength++;

    if (strength < 2) {
      strengthIndicator.className = "strength-indicator weak";
      strengthLabel = "Weak";
    } else if (strength < 4) {
      strengthIndicator.className = "strength-indicator medium";
      strengthLabel = "Medium";
    } else {
      strengthIndicator.className = "strength-indicator strong";
      strengthLabel = "Strong";
    }

    strengthText.textContent = `Password strength: ${strengthLabel}`;
  };

  passwordInput.addEventListener("input", updateStrength);

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    clearMessages();

    const firstName = document.getElementById("firstName").value.trim();
    const lastName = document.getElementById("lastName").value.trim();
    const email = document.getElementById("email").value.trim();
    const password = document.getElementById("password").value;
    const confirmPassword = document.getElementById("confirmPassword").value;
    const company = document.getElementById("company").value.trim();
    const terms = document.getElementById("terms").checked;

    if (!firstName || !lastName || !email || !password) {
      showError("All required fields must be filled.");
      return;
    }

    if (password !== confirmPassword) {
      showError("Passwords do not match.");
      return;
    }

    if (password.length < 8) {
      showError("Password must be at least 8 characters long.");
      return;
    }

    if (!terms) {
      showError("You must agree to the Terms of Service and Privacy Policy.");
      return;
    }

    toggleLoading(true);
    registerBtn.disabled = true;

    try {
      const formBody = `firstName=${encodeURIComponent(firstName)}&lastName=${encodeURIComponent(lastName)}&email=${encodeURIComponent(email)}&password=${encodeURIComponent(password)}&company=${encodeURIComponent(company)}`;

      const response = await fetch("/register", {
        method: "POST",
        headers: {
          "Content-Type": "application/x-www-form-urlencoded",
        },
        body: formBody,
      });

      const data = await response.json();

      if (response.ok) {
        showSuccess("Account created successfully! Redirecting to login in 2 seconds...");
        setTimeout(() => {
          window.location.href = "/login";
        }, 2000);
      } else if (response.status === 409) {
        showError("Email already registered. Please use a different email.");
      } else if (response.status === 400) {
        showError(data.message || "Registration failed. Please check your input.");
      } else {
        showError(data.message || "Registration failed. Please try again.");
      }
    } catch (error) {
      console.error("Error:", error);
      showError("An error occurred. Please try again later.");
    } finally {
      toggleLoading(false);
      registerBtn.disabled = false;
    }
  });
});
