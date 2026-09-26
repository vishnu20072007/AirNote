import { useEffect, useRef, useState } from "react";
import "./App.css";

function App() {
  const videoRef = useRef(null);
  const streamRef = useRef(null);

  const [cameraOn, setCameraOn] = useState(false);
  const [cameraError, setCameraError] = useState("");

  // =========================
  // START CAMERA
  // =========================
  const startCamera = async () => {
    try {
      setCameraError("");

      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: 1280,
          height: 720,
          facingMode: "user",
        },
        audio: false,
      });

      streamRef.current = stream;

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }

      setCameraOn(true);
    } catch (error) {
      console.error("Camera error:", error);
      setCameraError(
        "Camera access failed. Please allow camera permission."
      );
    }
  };

  // =========================
  // STOP CAMERA
  // =========================
  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }

    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }

    setCameraOn(false);
  };

  // Stop camera when page closes
  useEffect(() => {
    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
      }
    };
  }, []);

  return (
    <div className="app">

      {/* =========================
          HEADER
      ========================= */}
      <header className="header">
        <div>
          <h1>AirNote</h1>
          <p>Write. Draw. Save.</p>
        </div>
      </header>

      {/* =========================
          CAMERA SECTION
      ========================= */}
      <section className="camera-section">

        <div className="section-title">
          <div>
            <h2>Air Drawing</h2>
            <p>
              Use your hand in front of the camera to draw in the air.
            </p>
          </div>

          <div className="camera-status">
            <span
              className={`status-dot ${
                cameraOn ? "active" : ""
              }`}
            ></span>

            {cameraOn ? "Camera Active" : "Camera Off"}
          </div>
        </div>

        <div className="camera-container">

          <video
            ref={videoRef}
            className={`camera-video ${
              cameraOn ? "visible" : ""
            }`}
            autoPlay
            playsInline
            muted
          />

          {!cameraOn && (
            <div className="camera-placeholder">
              <div className="camera-icon">📷</div>

              <h3>Camera is Off</h3>

              <p>
                Start the camera to begin air drawing.
              </p>

              <button
                className="camera-btn"
                onClick={startCamera}
              >
                📷 Start Camera
              </button>
            </div>
          )}

          {/* Future hand tracking overlay */}
          {cameraOn && (
            <div className="tracking-overlay">
              <span className="tracking-text">
                Hand tracking ready
              </span>
            </div>
          )}

        </div>

        {cameraOn && (
          <button
            className="stop-camera-btn"
            onClick={stopCamera}
          >
            ⏹ Stop Camera
          </button>
        )}

        {cameraError && (
          <div className="camera-error">
            ⚠️ {cameraError}
          </div>
        )}

      </section>

      {/* =========================
          NEXT STEP INFO
      ========================= */}
      <section className="next-step">
        <div className="next-icon">✋</div>

        <div>
          <h3>Hand Tracking</h3>

          <p>
            Next, AirNote will detect your hand and finger
            position using MediaPipe.
          </p>
        </div>
      </section>

    </div>
  );
}

export default App;