# Hardware Integration Checklist

This document lists recommended hardware components, wiring notes, safety items, and a simple wiring/interaction diagram for integrating the Automated Visual Quality Inspection System into a production conveyor line.

## Overview
- Project is primarily software. Hardware required for line deployment:
  - Camera (GigE/USB3/CameraLink) with appropriate lens
  - Industrial PC or edge compute (Windows or Linux) running the software
  - Lighting (ring/line/structured) with stable power and diffusers
  - Conveyor with sensor (photo-eye, proximity) to trigger image capture
  - PLC (or I/O module) to receive reject signals (Modbus TCP or OPC-UA, or discrete relay)
  - Reject actuator (solenoid, air blast, pneumatic pusher) with safety interlocks
  - Optional frame-grabber hardware for Camera Link or CoaXPress

## Pre-deployment checklist

Hardware procurement
- Select camera model and lens: prefer global shutter for motion, monochrome sensor if imaging contrast is needed.
- Confirm interface: USB3/GigE/CameraLink and OS driver availability.
- Select illumination: ring or diffuse backlight depending on defect types.

Mounting & mechanics
- Rigid mount for camera and light; minimize vibrations.
- Ensure stable focal distance; use a macro lens for small features.
- Place diffuser/hooding to reduce ambient light.

Triggering & synchronization
- Place a conveyor sensor upstream of the inspection station at consistent distance.
- Configure sensor to trigger capture with predictable time-to-camera based on conveyor speed.
- For precise timing, use encoder pulses and hardware trigger where possible.

PLC / Network integration
- Choose protocol: Modbus TCP or OPC-UA recommended.
- Map reject output: coil/register or OPC node that the PLC will monitor.
- Ensure network segmentation and firewall rules allow PLC<->PC communication only over required ports.

Safety and interlocks
- Emergency stop integrated into the line and the PC's watchdog if available.
- Mechanical guards around reject actuator.
- Acknowledge PLC-based safety zones and follow local regulations.

Calibration & setup
- Use supplied `src/calibration.py` utilities to compute mm/px using a checkerboard or calibration target.
- Save calibration to the Streamlit app and verify measurements with known gauge blocks.
- Create and validate ROIs on the golden reference image in `app.py`.

MATLAB engine notes
- The project includes optional MATLAB `.m` helpers for sub-pixel boundary detection and alignment. These require the MATLAB engine for Python and corresponding toolboxes installed on the host.

## Minimal wiring diagram
Below is a simplified interaction diagram showing sensors, camera, PC, PLC and reject actuator.

```mermaid
flowchart LR
  Sensor[Conveyor Sensor\n(photo-eye)] -->|trigger| Camera[Camera (Global Shutter)]
  Camera -->|image| PC[Edge PC / Industrial PC\n(Runs Inspector)]
  PC -->|reject signal (Modbus/TCP or OPC-UA)| PLC[PLC / I/O Module]
  PLC -->|drive| Actuator[Reject Actuator\n(solenoid / pusher)]
  PLC -->|safety E-Stop| Safety[Safety Interlock]
  Lighting[Illumination] --> Camera
```

Notes:
- Replace the Modbus/OPC-UA arrow with a discrete relay output if your PLC expects hardware-level signals.
- For low-latency, hardware triggers (camera external trigger input) are preferred: sensor -> PLC or sensor -> trigger box -> camera.

## Recommended configuration parameters
- Camera exposure set to freeze motion at conveyor speed (use test images to tune).
- Use RAW/monochrome capture if available; avoid aggressive JPEG compression.
- Keep mm/px calibration tight: measure with multiple points across the field.

## On-site acceptance test (SAT)
1. Verify camera images are centered and in focus for all parts at production speed.
2. Run a batch of known-good parts — false rejects should be under your target rate.
3. Inject test defects and verify detection, classification, and reject actuation.
4. Validate SP C charts by producing a batch dataset and checking Cp/Cpk values in the Streamlit app.

## Troubleshooting
- If detections are noisy: check lighting, adjust CLAHE/filters in `src/inspector.py`.
- If reject timing is inconsistent: switch to encoder-based triggers or hardware trigger path.
- If MATLAB functions fail: ensure `matlab.engine` is installed and compatible with your Python interpreter.

---
For further assistance I can: provide a specific wiring diagram for your chosen PLC/camera model, or generate a printable checklist PDF. Which would you prefer?
