import { useState } from "react";
import "./App.css";

import Login from "./Login";
import Signup from "./Signup";

function App() {
  const [page, setPage] = useState("landing");

  // Check whether user is already logged in
  const [isLoggedIn, setIsLoggedIn] = useState(
    localStorage.getItem("airnoteLoggedIn") === "true"
  );

  // Login required popup
  const [showLoginPopup, setShowLoginPopup] = useState(false);

  // =========================
  // LOGIN PAGE
  // =========================
  if (page === "login") {
    return (
      <Login
        onSignup={() => setPage("signup")}
        onBack={() => setPage("landing")}
        onLoginSuccess={() => {
          localStorage.setItem("airnoteLoggedIn", "true");
          setIsLoggedIn(true);

          // After login, return to landing page for now.
          // Workspace will be connected later.
          setPage("landing");
        }}
      />
    );
  }

  // =========================
  // SIGNUP PAGE
  // =========================
  if (page === "signup") {
    return (
      <Signup
        onBackToLogin={() => setPage("login")}
      />
    );
  }

  // =========================
  // START WRITING
  // =========================
  const handleStartWriting = () => {
    if (isLoggedIn) {
      // Workspace will be connected here later.
      alert("You are logged in. Workspace will be connected here.");
    } else {
      setShowLoginPopup(true);
    }
  };

  // =========================
  // LANDING PAGE
  // =========================
  return (
    <div className="app">

      {/* =========================
          NAVBAR
      ========================= */}
      <header className="navbar">

        <div
          className="brand"
          onClick={() => setPage("landing")}
          style={{ cursor: "pointer" }}
        >
          <div className="brand-logo">
            A
          </div>

          <span>
            AirNote
          </span>
        </div>

        <nav className="nav-links">

          <a href="#features">
            Features
          </a>

          <a href="#about">
            About
          </a>

          <button
            className="login-nav-button"
            onClick={() => setPage("login")}
          >
            Log in
          </button>

        </nav>

      </header>


      {/* =========================
          MAIN
      ========================= */}
      <main>

        {/* =========================
            HERO
        ========================= */}
        <section className="hero-section">

          <div className="hero-left">

            <div className="hero-badge">

              <span className="badge-dot"></span>

              Write beyond the screen

            </div>


            <h1>

              Turn your hand

              <br />

              into{" "}

              <span className="gradient-text">
                pen.
              </span>

            </h1>


            <p className="hero-description">

              AirNote transforms your hand movements into digital
              writing. Create, draw, save and bring your ideas to life
              without touching the screen.

            </p>


            {/* HERO BUTTONS */}
            <div className="hero-buttons">

              {/* START WRITING */}

              <button
                className="primary-button"
                onClick={handleStartWriting}
              >

                Start Writing

                <span>
                  →
                </span>

              </button>


              {/* EXPLORE */}

              <a
                href="#features"
                className="secondary-button"
              >
                Explore AirNote
              </a>

            </div>


            <p className="powered-text">

              * Powered by computer vision

            </p>

          </div>


          {/* =========================
              HERO CANVAS PREVIEW
          ========================= */}
          <div className="hero-right">

            <div className="canvas-window">

              <div className="canvas-topbar">

                <div className="window-dots">

                  <span></span>
                  <span></span>
                  <span></span>

                </div>


                <strong>
                  AirNote Canvas
                </strong>


                <div className="live-status">

                  <span></span>

                  LIVE

                </div>

              </div>


              <div className="canvas-area">

                <div className="camera-card">

                  <div className="camera-icon">
                    CAM
                  </div>

                  <div>

                    <strong>
                      Camera Input
                    </strong>

                    <small>
                      Ready to capture
                    </small>

                  </div>

                </div>


                <svg
                  className="drawing-line"
                  viewBox="0 0 700 300"
                  preserveAspectRatio="none"
                >

                  <path
                    d="
                      M70 190
                      C120 100, 150 230, 210 155
                      C260 90, 290 70, 330 170
                      C370 275, 430 270, 475 145
                      C520 25, 575 50, 635 135
                    "
                  />

                </svg>


                <div className="cursor-dot"></div>


                <div className="pinch-card">

                  <strong>
                    PINCH
                  </strong>

                  <span>
                    Pinch to write
                  </span>

                </div>


                <div className="autosave-card">

                  <div className="save-icon">
                    OK
                  </div>

                  <div>

                    <strong>
                      Auto saved
                    </strong>

                    <small>
                      Just now
                    </small>

                  </div>

                </div>

              </div>


              <div className="canvas-footer">

                <span>
                  Drawing
                </span>

                <span>
                  Stroke 02
                </span>

              </div>

            </div>

          </div>

        </section>


        {/* =========================
            FEATURES
        ========================= */}
        <section
          id="features"
          className="features-section"
        >

          <div className="feature-card">

            <span className="feature-number">
              01
            </span>

            <h3>
              Air Writing
            </h3>

            <p>
              Write naturally with your hand.
            </p>

          </div>


          <div className="feature-card">

            <span className="feature-number">
              02
            </span>

            <h3>
              Smart Canvas
            </h3>

            <p>
              Capture every stroke digitally.
            </p>

          </div>


          <div className="feature-card">

            <span className="feature-number">
              03
            </span>

            <h3>
              Save and Export
            </h3>

            <p>
              Keep your ideas whenever you need them.
            </p>

          </div>

        </section>


        {/* =========================
            ABOUT
        ========================= */}
        <section
          id="about"
          className="about-section"
        >

          <div className="about-content">

            <span className="section-label">
              ABOUT AIRNOTE
            </span>


            <h2>

              Your hands.

              <br />

              Your ideas.

              <br />

              <span className="gradient-text">
                No screen required.
              </span>

            </h2>


            <p>

              AirNote uses computer vision to understand your
              hand movements and transform them into digital
              strokes. It gives you a natural way to write and
              create without physically touching your screen.

            </p>

          </div>

        </section>

      </main>


      {/* =========================
          LOGIN REQUIRED POPUP
      ========================= */}
      {showLoginPopup && (

        <div className="login-popup-overlay">

          <div className="login-popup">

            <div className="popup-icon">
              A
            </div>


            <h2>
              Please log in first
            </h2>


            <p>
              You need to log in before you can start writing
              with AirNote.
            </p>


            <div className="popup-actions">

              <button
                className="popup-cancel"
                onClick={() => setShowLoginPopup(false)}
              >
                Cancel
              </button>


              <button
                className="popup-login"
                onClick={() => {
                  setShowLoginPopup(false);
                  setPage("login");
                }}
              >
                Go to Login
              </button>

            </div>

          </div>

        </div>

      )}

    </div>
  );
}

export default App;