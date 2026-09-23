import "./Workspace.css";

function Workspace({ onLogout }) {
  return (
    <div className="workspace">

      {/* HEADER */}
      <header className="workspace-header">
        <div className="workspace-brand">
          <div className="workspace-logo">A</div>
          <span>AirNote</span>
        </div>

        <div className="workspace-actions">
          <span className="workspace-status">
            ● Ready
          </span>

          <button onClick={onLogout} className="logout-button">
            Log out
          </button>
        </div>
      </header>


      {/* MAIN WORKSPACE */}
      <main className="workspace-main">

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


        {/* WORK AREA */}
        <section className="work-area">

          {/* CAMERA */}
          <div className="panel camera-panel">

            <div className="panel-header">
              <div>
                <span className="panel-label">
                  CAMERA INPUT
                </span>
                <h2>Camera Preview</h2>
              </div>

              <span className="live-badge">
                LIVE
              </span>
            </div>

            <div className="camera-preview">

              <div className="camera-placeholder">
                <div className="camera-icon-large">
                  CAM
                </div>

                <h3>Camera Ready</h3>

                <p>
                  Your camera preview will appear here.
                </p>
              </div>

            </div>

          </div>


          {/* CANVAS */}
          <div className="panel canvas-panel">

            <div className="panel-header">
              <div>
                <span className="panel-label">
                  DIGITAL CANVAS
                </span>
                <h2>AirNote Canvas</h2>
              </div>

              <span className="stroke-count">
                Stroke 00
              </span>
            </div>

            <div className="drawing-canvas">

              <div className="canvas-message">
                <span>✦</span>
                <p>
                  Pinch your fingers to start writing
                </p>
              </div>

            </div>

          </div>

        </section>


        {/* TOOLBAR */}
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


        {/* BOTTOM INFO */}
        <section className="workspace-info">

          <div>
            <span>GESTURE</span>
            <strong>Pinch to Write</strong>
          </div>

          <div>
            <span>STATUS</span>
            <strong>Waiting for camera</strong>
          </div>

          <div>
            <span>AUTOSAVE</span>
            <strong>Enabled</strong>
          </div>

        </section>

      </main>

    </div>
  );
}

export default Workspace;