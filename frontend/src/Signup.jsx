import { useState } from "react";
import "./Signup.css";

function Signup({ onBackToLogin }) {
  const [step, setStep] = useState(1);

  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [otp, setOtp] = useState("");

  const [username, setUsername] = useState("");
  const [newPassword, setNewPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");

  const handleSendOTP = (e) => {
    e.preventDefault();

    if (!email || !password) {
      alert("Please enter your Gmail and password.");
      return;
    }

    // Temporary frontend OTP simulation
    alert("OTP sent to your Gmail.");

    setStep(2);
  };

  const handleVerifyOTP = (e) => {
    e.preventDefault();

    if (!otp) {
      alert("Please enter the OTP.");
      return;
    }

    // Temporary OTP verification
    setStep(3);
  };

  const handleCreateAccount = (e) => {
    e.preventDefault();

    if (!username || !newPassword || !confirmPassword) {
      alert("Please fill all fields.");
      return;
    }

    if (newPassword !== confirmPassword) {
      alert("Passwords do not match.");
      return;
    }

    // Save temporarily in browser
    localStorage.setItem(
      "airnoteUser",
      JSON.stringify({
        email,
        username,
        password: newPassword,
      })
    );

    alert("Account created successfully!");

    onBackToLogin();
  };

  return (
    <div className="signup-page">

      <div className="signup-card">

        {/* BRAND */}
        <div className="signup-brand">
          <div className="signup-logo">A</div>
          <div>AirNote</div>
        </div>

        {/* STEP 1 */}
        {step === 1 && (
          <>
            <h1>Create your account</h1>

            <p className="signup-subtitle">
              Start writing beyond the screen.
            </p>

            <form onSubmit={handleSendOTP}>

              <label>Gmail</label>

              <input
                type="email"
                placeholder="Enter your Gmail"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
              />

              <label>Password</label>

              <input
                type="password"
                placeholder="Create a password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
              />

              <button className="signup-primary">
                Continue
              </button>

            </form>

            <p className="login-link">
              Already have an account?
              <button onClick={onBackToLogin}>
                Log in
              </button>
            </p>
          </>
        )}

        {/* STEP 2 */}
        {step === 2 && (
          <>
            <h1>Verify your Gmail</h1>

            <p className="signup-subtitle">
              Enter the OTP sent to
              <br />
              <strong>{email}</strong>
            </p>

            <form onSubmit={handleVerifyOTP}>

              <label>OTP</label>

              <input
                type="text"
                maxLength="6"
                placeholder="Enter 6-digit OTP"
                value={otp}
                onChange={(e) => setOtp(e.target.value)}
              />

              <button className="signup-primary">
                Verify OTP
              </button>

            </form>

            <button
              className="back-link"
              onClick={() => setStep(1)}
            >
              ← Back
            </button>
          </>
        )}

        {/* STEP 3 */}
        {step === 3 && (
          <>
            <h1>Create your profile</h1>

            <p className="signup-subtitle">
              Choose your AirNote username and password.
            </p>

            <form onSubmit={handleCreateAccount}>

              <label>Username</label>

              <input
                type="text"
                placeholder="Choose a username"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
              />

              <label>New Password</label>

              <input
                type="password"
                placeholder="Enter new password"
                value={newPassword}
                onChange={(e) => setNewPassword(e.target.value)}
              />

              <label>Confirm New Password</label>

              <input
                type="password"
                placeholder="Confirm new password"
                value={confirmPassword}
                onChange={(e) => setConfirmPassword(e.target.value)}
              />

              <button className="signup-primary">
                Create Account
              </button>

            </form>

            <button
              className="back-link"
              onClick={() => setStep(2)}
            >
              ← Back
            </button>
          </>
        )}

      </div>

    </div>
  );
}

export default Signup;