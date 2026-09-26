import { useEffect, useRef, useState } from "react";
import "./Workspace.css";

function Workspace({ onLogout }) {
  const videoRef = useRef(null);
  const streamRef = useRef(null);

  const [cameraOn, setCameraOn] = useState(false);
  const [cameraError, setCameraError] = useState("");
  const [status, setStatus] = useState("Waiting for camera");

  // =========================
  // START CAMERA
  // =========================
  const startCamera = async () => {
    try {
      setCameraError("");
      setStatus("Starting camera...");

      if (!navigator.mediaDevices?.getUserMedia) {
        setCameraError(
          "Camera access is not supported by this browser."
        );
        setStatus("Camera unavailable");
        return;
      }

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

        await videoRef.current.play().catch(() => {});
      }

      setCameraOn(true);
      setStatus("Camera active");
    } catch (error) {
      console.error("Camera error:", error);

      setCameraError(
        "Camera access failed. Please allow camera permission."
      );

      setStatus("Camera permission required");
      setCameraOn(false);
    }
  };

  // =========================
  // STOP CAMERA
  // =========================
  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => {
        track.stop();
      });

      streamRef.current = null;
    }

    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }

    setCameraOn(false);
    setStatus("Waiting for camera");
  };

  // =========================
  // CLEANUP
  // =========================
  useEffect(() => {
    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => {
          track.stop();
        });

        streamRef.current = null;
      }
    };
  }, []);

  // =========================
  // LOGOUT
  // =========================
  const handleLogout = () => {
    stopCamera();
    onLogout();
  };

  return (
    <div className="workspace">

      {/* =========================
          HEADER
      ========================= */}
      <header className="workspace-header">

        <div className="workspace-brand">
          <div className="workspace-logo">
            A
          </div>

          <span>
            AirNote
          </span>
        </div>


        <div className="workspace-actions">

          <span className="workspace-status">

            <span
              className={`status-indicator ${
                cameraOn ? "active" : ""
              }`}
            ></span>

            {cameraOn
              ? "Camera Active"
              : "Ready"}

          </span>


          <button
            onClick={handleLogout}
            className="logout-button"
          >
            Log out
          </button>

        </div>

      </header>


      {/* =========================
          MAIN WORKSPACE
      ========================= */}
      <main className="workspace-main">

        {/* TITLE */}
        <div className="workspace-title">

          <div>

            <span className="workspace-label">
              AIRNOTE WORKSPACE
            </span>

            <h1>
              Write beyond the screen.
            </h1>

            <p>
              Use your hand movements to create digital strokes.
            </p>

          </div>

        </div>


        {/* =========================
            WORK AREA
        ========================= */}
        <section className="work-area">


          {/* =========================
              CAMERA PANEL
          ========================= */}
          <div className="panel camera-panel">

            <div className="panel-header">

              <div>

                <span className="panel-label">
                  CAMERA INPUT
                </span>

                <h2>
                  Camera Preview
                </h2>

              </div>


              <span
                className={`live-badge ${
                  cameraOn ? "camera-active" : ""
                }`}
              >
                {cameraOn ? "LIVE" : "OFF"}
              </span>

            </div>


            {/* CAMERA PREVIEW */}
            <div className="camera-preview">

              {cameraOn ? (

                <video
                  ref={videoRef}
                  className="camera-video"
                  autoPlay
                  playsInline
                  muted
                />

              ) : (

                <div className="camera-placeholder">

                  <div className="camera-icon-large">
                    CAM
                  </div>

                  <h3>
                    Camera Ready
                  </h3>

                  <p>
                    Start the camera to begin air drawing.
                  </p>


                  <button
                    className="camera-start-button"
                    onClick={startCamera}
                  >
                    Start Camera
                  </button>

                </div>

              )}

            </div>


            {/* STOP CAMERA */}
            {cameraOn && (

              <button
                className="camera-stop-button"
                onClick={stopCamera}
              >
                Stop Camera
              </button>

            )}


            {/* ERROR */}
            {cameraError && (

              <div className="camera-error">
                ⚠ {cameraError}
              </div>

            )}

          </div>


          {/* =========================
              DIGITAL CANVAS
          ========================= */}
          <div className="panel canvas-panel">

            <div className="panel-header">

              <div>

                <span className="panel-label">
                  DIGITAL CANVAS
                </span>

                <h2>
                  AirNote Canvas
                </h2>

              </div>


              <span className="stroke-count">
                Stroke 00
              </span>

            </div>


            <div className="drawing-canvas">

              <div className="canvas-message">

                <span>
                  ✦
                </span>

                <p>
                  Pinch your fingers to start writing
                </p>

              </div>

            </div>

          </div>

        </section>


        {/* =========================
            TOOLBAR
        ========================= */}
        <section className="toolbar">

          <button className="tool-button active">
            ✎ Draw
          </button>

          <button className="tool-button">
            ◇ Eraser
          </button>

          <button className="tool-button">
            ↶ Undo
          </button>

          <button className="tool-button">
            ↷ Redo
          </button>

          <button className="tool-button">
            Clear
          </button>


          <div className="toolbar-spacer"></div>


          <button className="save-button">
            ↓ Save
          </button>

          <button className="export-button">
            Export
          </button>

        </section>


        {/* =========================
            BOTTOM INFO
        ========================= */}
        <section className="workspace-info">

          <div>

            <span>
              GESTURE
            </span>

            <strong>
              Pinch to Write
            </strong>

          </div>


          <div>

            <span>
              STATUS
            </span>

            <strong>
              {status}
            </strong>

          </div>


          <div>

            <span>
              AUTOSAVE
            </span>

            <strong>
              Enabled
            </strong>

          </div>

        </section>

      </main>

    </div>
  );
}

export default Workspace;