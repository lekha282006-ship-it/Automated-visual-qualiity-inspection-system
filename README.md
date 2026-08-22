# Automated Visual Quality Inspection System

This repository contains a modular industrial visual inspection system with real-time SPC and optional MATLAB integration. It's designed for development and prototyping of machine-vision inspection pipelines and SPC analytics.

## Repository Layout
- `app.py` - Streamlit dashboard (calibration, single inspection, batch processing, SPC).
- `webcam_live.py` - Conveyor simulator / live inspection runner.
- `src/` - Core modules:
  - `inspector.py` - OpenCV inspection engine (alignment, defect segmentation, ROI support, MATLAB hooks).
  - `calibration.py` - Checkerboard camera calibration (mm/px conversion helpers).
  - `db.py` - SQLite persistence for inspection logs.
  - `spc.py` - SPC engine (Cp/Cpk/Pp/Ppk, control limits, WECO rules).
  - `camera.py` - Threaded video capture helper.
  - `conveyor_sim.py` - Conveyor simulator with photo-eye trigger.
- Hardware integration checklist and wiring guidance: see `docs/HARDWARE.md`
- `matlab/` - MATLAB helper scripts (`spc_analysis.m`, `inspect_part.m`) for teams using MATLAB.
- `tests/` - Unit tests (`pytest`).

## Quickstart (Python)
1. Create and activate a virtual environment (recommended):

```bash
python -m venv .venv
.venv\Scripts\activate    # Windows
pip install -r requirements.txt
```

2. Run unit tests:

```bash
python -m pytest -q
```

3. Launch the Streamlit dashboard:

```bash
python -m streamlit run app.py --server.port 8501 --server.headless true
```

4. Run the conveyor simulator (webcam required):

```bash
python webcam_live.py
```

## MATLAB Integration
- `matlab/spc_analysis.m` and `matlab/inspect_part.m` are provided as reference MATLAB implementations of the SPC analytics and an example inspection routine.
- To call MATLAB from Python, install MATLAB and enable `matlab.engine` for Python. Python-side hooks are present in `src/inspector.py` and `src/calibration.py` as placeholders.

## Running Calibration
- Use `src/calibration.py` utilities for checkerboard camera calibration to compute `mm_per_px`. Save and pass the calibration to the Streamlit app or inspector.

## Docker (optional)
- Build the image:

```bash
docker build -t avqi-system .
```

- Run the container (exposes Streamlit 8501):

```bash
docker run -p 8501:8501 avqi-system
```

## Deliverables
- Modular Python codebase under `src/`.
- MATLAB helper scripts in `matlab/`.
- Unit tests in `tests/` (run with `pytest`).
- `requirements.txt` and `Dockerfile` for deployment.

If you want, I can:
- Add interactive ROI drawing in the Streamlit UI using `streamlit-drawable-canvas`.
- Integrate Modbus/OPC-UA for live reject signals.
- Implement full MATLAB <-> Python handoff examples with `matlab.engine`.

Open a task and I'll implement it next.
