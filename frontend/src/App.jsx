import { useEffect, useRef, useState } from "react";
import "./App.css";

function App() {
  // =========================
  // CAMERA
  // =========================
  const videoRef = useRef(null);
  const streamRef = useRef(null);

  // =========================
  // CANVAS
  // =========================
  const canvasRef = useRef(null);
  const drawingRef = useRef(false);
  const lastPointRef = useRef(null);

  // =========================
  // STATE
  // =========================
  const [cameraOn, setCameraOn] = useState(false);
  const [cameraError, setCameraError] = useState("");
  const [strokes, setStrokes] = useState([]);
  const [status, setStatus] = useState("Ready");

  // =========================
  // START CAMERA
  // =========================
  const startCamera = async () => {
    try {
      setCameraError("");

      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 1280 },
          height: { ideal: 720 },
          facingMode: "user",
        },
        audio: false,
      });

      streamRef.current = stream;

      setCameraOn(true);

      // Attach stream after React renders video
      setTimeout(async () => {
        if (videoRef.current) {
          videoRef.current.srcObject = stream;

          try {
            await videoRef.current.play();
            setStatus("Camera Active");
          } catch (error) {
            console.error("Video play error:", error);
          }
        }
      }, 100);

    } catch (error) {
      console.error("Camera error:", error);

      setCameraError(
        "Camera access failed. Please allow camera permission."
      );

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
      videoRef.current.pause();
      videoRef.current.srcObject = null;
    }

    setCameraOn(false);
    setStatus("Camera Off");
  };

  // =========================
  // CLEAN CAMERA
  // =========================
  useEffect(() => {
    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => {
          track.stop();
        });
      }
    };
  }, []);

  // =========================
  // CANVAS SETUP
  // =========================
  useEffect(() => {
    const canvas = canvasRef.current;

    if (!canvas) return;

    const ctx = canvas.getContext("2d");

    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    ctx.lineWidth = 4;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.strokeStyle = "#111827";
  }, []);

  // =========================
  // GET CANVAS POSITION
  // =========================
  const getCanvasPoint = (event) => {
    const canvas = canvasRef.current;

    const rect = canvas.getBoundingClientRect();

    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;

    return {
      x: (event.clientX - rect.left) * scaleX,
      y: (event.clientY - rect.top) * scaleY,
    };
  };

  // =========================
  // START DRAWING
  // =========================
  const startDrawing = (event) => {
    const point = getCanvasPoint(event);

    drawingRef.current = true;
    lastPointRef.current = point;

    canvasRef.current.setPointerCapture(event.pointerId);
  };

  // =========================
  // DRAW
  // =========================
  const draw = (event) => {
    if (!drawingRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");

    const point = getCanvasPoint(event);
    const last = lastPointRef.current;

    if (!last) {
      lastPointRef.current = point;
      return;
    }

    ctx.beginPath();

    ctx.moveTo(last.x, last.y);
    ctx.lineTo(point.x, point.y);

    ctx.stroke();

    setStrokes((previous) => [
      ...previous,
      {
        x1: last.x,
        y1: last.y,
        x2: point.x,
        y2: point.y,
      },
    ]);

    lastPointRef.current = point;
  };

  // =========================
  // STOP DRAWING
  // =========================
  const stopDrawing = () => {
    drawingRef.current = false;
    lastPointRef.current = null;
  };

  // =========================
  // CLEAR CANVAS
  // =========================
  const clearCanvas = () => {
    const canvas = canvasRef.current;

    if (!canvas) return;

    const ctx = canvas.getContext("2d");

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    ctx.lineWidth = 4;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.strokeStyle = "#111827";

    setStrokes([]);
    setStatus("Canvas Cleared");
  };

  // =========================
  // SAVE DRAWING TO BACKEND
  // =========================
  const saveDrawing = async () => {
    try {
      setStatus("Saving...");

      const response = await fetch(
        "http://127.0.0.1:8000/drawings",
        {
          method: "POST",

          headers: {
            "Content-Type": "application/json",
            Accept: "application/json",
          },

          body: JSON.stringify({
            name: "AirNote Drawing",
            stroke_data: JSON.stringify(strokes),
          }),
        }
      );

      if (!response.ok) {
        throw new Error("Failed to save drawing");
      }

      const data = await response.json();

      console.log("Saved drawing:", data);

      setStatus("Drawing Saved ✓");

    } catch (error) {
      console.error("Save error:", error);

      setStatus("Save Failed");

      alert(
        "Could not save drawing. Make sure the AirNote backend is running."
      );
    }
  };

  // =========================
  // UI
  // =========================
  return (
    <div className="app">

      {/* =========================
          HEADER
      ========================= */}
      <header className="header">
        <h1>AirNote</h1>
        <p>Write. Draw. Save.</p>
      </header>


      {/* =========================
          WORKSPACE
      ========================= */}
      <section className="workspace">

        <h2>AirNote Workspace</h2>

        <p>
          Use the camera for hand tracking and draw inside the workspace.
        </p>


        {/* =========================
            CONTROLS
        ========================= */}
        <div className="workspace-controls">

          {!cameraOn ? (
            <button
              className="camera-btn"
              onClick={startCamera}
            >
              📷 Start Camera
            </button>
          ) : (
            <button
              className="stop-camera-btn"
              onClick={stopCamera}
            >
              ⏹ Stop Camera
            </button>
          )}

          <button onClick={clearCanvas}>
            🗑 Clear
          </button>

          <button onClick={saveDrawing}>
            💾 Save
          </button>

        </div>


        {/* =========================
            STATUS
        ========================= */}
        <div className="camera-status">

          <span
            className={`status-dot ${
              cameraOn ? "active" : ""
            }`}
          ></span>

          {cameraOn ? "Camera Active" : "Camera Off"}

        </div>


        {/* =========================
            CAMERA
        ========================= */}
        <div className="camera-wrapper">

          <h2>Camera</h2>

          <div className="camera-container">

            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              className="camera-video"
            />

            {!cameraOn && (
              <div className="camera-placeholder">

                <div className="camera-icon">
                  📷
                </div>

                <h3>Camera is Off</h3>

                <p>
                  Start the camera to begin.
                </p>

                <button
                  className="camera-btn"
                  onClick={startCamera}
                >
                  Start Camera
                </button>

              </div>
            )}

            {cameraOn && (
              <div className="tracking-overlay">

                <span>
                  ✋ Hand tracking ready
                </span>

              </div>
            )}

          </div>

        </div>


        {/* =========================
            ERROR
        ========================= */}
        {cameraError && (
          <div className="camera-error">
            ⚠️ {cameraError}
          </div>
        )}


        {/* =========================
            DRAWING CANVAS
        ========================= */}
        <div className="drawing-section">

          <h2>Drawing Canvas</h2>

          <p>
            Draw here using your mouse or touch.
            Hand-pinch drawing will be connected next.
          </p>

          <canvas
            ref={canvasRef}
            width={1100}
            height={500}
            className="drawing-canvas"

            onPointerDown={startDrawing}
            onPointerMove={draw}
            onPointerUp={stopDrawing}
            onPointerCancel={stopDrawing}
            onPointerLeave={stopDrawing}
          />

        </div>


        {/* =========================
            STATUS
        ========================= */}
        <div className="drawing-status">
          <strong>Status:</strong> {status}
        </div>

      </section>

    </div>
  );
}

export default App;