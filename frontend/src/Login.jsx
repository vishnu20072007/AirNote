import { useState } from "react";
import "./Login.css";

function Login({
  onSignup,
  onBack,
  onLoginSuccess
}) {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");

  const handleLogin = (e) => {
    e.preventDefault();

    // Get registered account
    const savedUser = JSON.parse(
      localStorage.getItem("airnoteUser")
    );

    // Check credentials
    if (
      savedUser &&
      username === savedUser.username &&
      password === savedUser.password
    ) {
      alert("Login successful!");

      // Go to AirNote Workspace
      onLoginSuccess();
    } else {
      alert("Invalid username or password.");
    }
  };


  const handleGoogleLogin = () => {
    alert("Google login will be connected later.");
  };


  return (
    <div className="login-page">

      <div className="login-card">

        {/* LOGO */}
        <div className="login-brand">

          <div className="login-logo">
            A
          </div>

          <span>
            AirNote
          </span>

        </div>


        {/* TITLE */}
        <h1>
          Welcome back
        </h1>

        <p className="login-subtitle">
          Sign in to continue writing beyond the screen.
        </p>


        {/* LOGIN FORM */}
        <form onSubmit={handleLogin}>

          {/* USERNAME */}
          <div className="form-group">

            <label>
              Username
            </label>

            <input
              type="text"
              placeholder="Enter your username"
              value={username}
              onChange={(e) =>
                setUsername(e.target.value)
              }
              required
            />

          </div>


          {/* PASSWORD */}
          <div className="form-group">

            <label>
              Password
            </label>

            <input
              type="password"
              placeholder="Enter your password"
              value={password}
              onChange={(e) =>
                setPassword(e.target.value)
              }
              required
            />

          </div>


          {/* FORGOT PASSWORD */}
          <div className="forgot-wrapper">

            <button
              type="button"
              className="forgot-password"
              onClick={() =>
                alert("Password recovery will be added later.")
              }
            >
              Forgot password?
            </button>

          </div>


          {/* LOGIN BUTTON */}
          <button
            type="submit"
            className="login-submit"
          >
            Log in
          </button>

        </form>


        {/* OR */}
        <div className="or-divider">

          <span></span>

          <p>OR</p>

          <span></span>

        </div>


        {/* GOOGLE */}
        <button
          type="button"
          className="google-button"
          onClick={handleGoogleLogin}
        >

          <span className="google-icon">
            G
          </span>

          Continue with Google

        </button>


        {/* SIGNUP */}
        <div className="signup-link">

          <span>
            Don't have an account?
          </span>

          <button
            type="button"
            onClick={onSignup}
          >
            Sign up
          </button>

        </div>


        {/* BACK */}
        <button
          type="button"
          className="back-button"
          onClick={onBack}
        >
          ← Back to AirNote
        </button>

      </div>

    </div>
  );
}

export default Login;