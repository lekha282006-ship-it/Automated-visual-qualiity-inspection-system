import streamlit as st
import cv2
import numpy as np
import os
import tempfile
from src.inspector import PartInspector
from src.spc import SPCEngine
from src.db import DBEngine
from src.logging_config import setup_logging, get_logger
from src.metrics import start_metrics_server, inspections_total, inspections_failed, inspection_duration
import plotly.express as px
import plotly.graph_objects as go
from src import calibration
from streamlit_drawable_canvas import st_canvas
import json
from PIL import Image

st.set_page_config(layout="wide", page_title="Industrial Vision SPC")
st.title("Industrial Vision Inspector with Real-Time SPC")

# setup logging & metrics
logger = setup_logging()
log = get_logger('app')
start_metrics_server()

# Simple Streamlit auth: optional password via environment variable STREAMLIT_PASSWORD
if "auth" not in st.session_state:
    st.session_state["auth"] = False

PASSWORD = os.environ.get('STREAMLIT_PASSWORD')
if PASSWORD:
    if not st.session_state["auth"]:
        with st.sidebar.expander("Login", expanded=True):
            pw = st.text_input("App Password", type="password")
            if st.button("Login"):
                if pw == PASSWORD:
                    st.session_state["auth"] = True
                    st.experimental_rerun()
                else:
                    st.error("Invalid password")
    if not st.session_state["auth"]:
        st.stop()

# Initialize session state for batch results
if "batch_results" not in st.session_state:
    st.session_state["batch_results"] = []
if "inspector" not in st.session_state:
    st.session_state["inspector"] = None
if "db" not in st.session_state:
    st.session_state["db"] = DBEngine("inspections.db")
if "calib" not in st.session_state:
    st.session_state["calib"] = None
if "ref_img_bgr" not in st.session_state:
    st.session_state["ref_img_bgr"] = None
if "rois" not in st.session_state:
    st.session_state["rois"] = {}

st.sidebar.header("System Calibration")
with st.sidebar.expander("Reference & Calibration", expanded=True):
    ref_file = st.file_uploader("Upload Golden Reference", type=["png", "jpg"])
    size_tol = st.slider("Dimensional Tolerance (%)", 1.0, 20.0, 5.0)
    defect_thresh = st.slider("Surface Defect Threshold (px)", 10, 500, 100)
    mm_per_px = st.number_input("mm per pixel (optional)", min_value=0.0, value=0.0, step=0.01)

    if ref_file is not None:
        tfile_ref = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        tfile_ref.write(ref_file.read())
        tfile_ref.close()
        try:
            img = cv2.imread(tfile_ref.name)
            st.session_state["ref_img_bgr"] = img
            st.session_state["inspector"] = PartInspector(tfile_ref.name, size_tol, defect_thresh)
            # apply calibration to inspector if provided
            if mm_per_px > 0:
                try:
                    st.session_state["inspector"].set_calibration(float(mm_per_px))
                except Exception:
                    pass
            # if user provided mm_per_px store as calibration
            if mm_per_px > 0:
                st.session_state["calib"] = {"mm_per_px": float(mm_per_px)}
            st.success("Golden Reference Calibrated.")
        except Exception as e:
            st.error(f"Calibration Error: {e}")
        finally:
            os.unlink(tfile_ref.name)

    if st.session_state.get("ref_img_bgr") is not None:
        st.image(cv2.cvtColor(st.session_state["ref_img_bgr"], cv2.COLOR_BGR2RGB), caption="Current Golden Reference")

    if st.button("Run full checkerboard calibration"):
        st.info("See `src/calibration.py` for calibration utilities. Run offline and provide mm/px.")

with st.sidebar.expander("Industrial Network", expanded=False):
    if "net_config" not in st.session_state:
        st.session_state["net_config"] = {"protocol": "modbus", "endpoint": "127.0.0.1:502", "username": "", "password": ""}

    proto = st.selectbox("Protocol", options=["modbus", "opcua"], index=0)
    endpoint = st.text_input("Endpoint (host:port or URL)", value=st.session_state["net_config"].get("endpoint", "127.0.0.1:502"))
    if proto == "opcua":
        user = st.text_input("OPC-UA Username", value=st.session_state["net_config"].get("username", ""))
        pwd = st.text_input("OPC-UA Password", value=st.session_state["net_config"].get("password", ""), type="password")
    else:
        user = ""
        pwd = ""

    if st.button("Save Network Config"):
        st.session_state["net_config"] = {"protocol": proto, "endpoint": endpoint, "username": user, "password": pwd}
        st.success("Saved network configuration.")

    if st.button("Test Connection"):
        cfg = st.session_state.get("net_config", {})
        try:
            from src.industrial_net import create_client

            client = create_client(cfg.get("protocol"), cfg.get("endpoint"), username=cfg.get("username"), password=cfg.get("password"))
            # attempt a simple connect and send_reject (non-destructive)
            try:
                client.connect()
                client.send_reject({"timestamp": 0.0})
                client.close()
                st.success("Connection test succeeded (or stubbed).")
            except Exception as e:
                st.error(f"Connection test failed: {e}")
        except Exception as e:
            st.error(f"Could not create client: {e}")

if st.session_state["inspector"] is not None:
    tab1, tab2, tab3 = st.tabs(["Interactive Inspection", "Batch Processing", "SPC Dashboard"])
    
    with tab1:
        st.header("Single Part Inspection")
        st.subheader("ROI Management")
        if st.session_state.get("ref_img_bgr") is None:
            st.info("Upload a Golden Reference in the sidebar to enable ROI tools.")
        else:
            img = st.session_state["ref_img_bgr"].copy()
            h, w = img.shape[:2]
            st.subheader("Interactive ROI Drawing")
            st.write("Use the canvas to draw rectangles or polygons on the reference image and save them as ROIs.")
            # prepare background image for canvas
            bg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            bg_pil = Image.fromarray(bg)
            canvas_result = st_canvas(
                fill_color="rgba(255, 0, 0, 0.3)",  # fill color with some opacity
                stroke_width=2,
                stroke_color="#ff0000",
                background_image=bg_pil,
                update_streamlit=True,
                height=h,
                width=w,
                drawing_mode="polygon",
                key="ref_canvas",
            )
            if canvas_result.json_data is not None:
                objects = canvas_result.json_data.get("objects", [])
                st.write(f"Detected {len(objects)} shapes on canvas.")
                if objects:
                    roi_save_name = st.text_input("Save ROI name", value="canvas_roi")
                    roi_save_tol = st.number_input("ROI tol (fraction)", min_value=0.0, max_value=1.0, value=0.02)
                    if st.button("Save ROI from Canvas"):
                        mask = np.zeros((h, w), dtype=np.uint8)
                        for obj in objects:
                            if obj.get("type") == "polygon" or obj.get("type") == "path":
                                points = obj.get("path") or obj.get("points") or []
                                # path comes as list of [x,y] pairs
                                pts = []
                                for p in points:
                                    if isinstance(p, list) and len(p) >= 2:
                                        pts.append([int(p[0]), int(p[1])])
                                if pts:
                                    cv2.fillPoly(mask, [np.array(pts, dtype=np.int32)], 255)
                            elif obj.get("type") == "rect":
                                left = int(obj.get("left", 0))
                                top = int(obj.get("top", 0))
                                width_rect = int(obj.get("width", 0))
                                height_rect = int(obj.get("height", 0))
                                cv2.rectangle(mask, (left, top), (left + width_rect, top + height_rect), 255, -1)
                            st.session_state["inspector"].add_roi(roi_save_name, mask, roi_save_tol)
                            # persist ROI mask in session for UI management
                            st.session_state["rois"][roi_save_name] = {"mask": mask, "tol": float(roi_save_tol)}
                            st.success(f"Saved ROI '{roi_save_name}' from canvas")
                            st.experimental_rerun()
            st.write("Define ROI in normalized coordinates (0..1)")
            colA, colB = st.columns(2)
            with colA:
                rx = st.slider("x0 (left)", 0.0, 1.0, 0.6)
                ry = st.slider("y0 (top)", 0.0, 1.0, 0.3)
            with colB:
                rx2 = st.slider("x1 (right)", 0.0, 1.0, 0.95)
                ry2 = st.slider("y1 (bottom)", 0.0, 1.0, 0.6)
            roi_name = st.text_input("ROI name", value="mount_hole")
            roi_tol = st.number_input("ROI tolerance (fraction)", min_value=0.0, max_value=1.0, value=0.02)
            # draw overlay
            x0 = int(rx * w); y0 = int(ry * h); x1 = int(rx2 * w); y1 = int(ry2 * h)
            overlay = img.copy()
            cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 255, 0), 2)
            st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), caption="Reference with ROI preview")
            if st.button("Add ROI to Inspector"):
                mask = np.zeros((h, w), dtype=np.uint8)
                mask[y0:y1, x0:x1] = 255
                st.session_state["inspector"].add_roi(roi_name, mask, roi_tol)
                st.session_state["rois"][roi_name] = {"mask": mask, "tol": float(roi_tol)}
                st.success(f"Added ROI '{roi_name}'")
                st.experimental_rerun()

        # ROI Management list
        st.subheader("Saved ROIs")
        if st.session_state["rois"]:
            for name, meta in list(st.session_state["rois"].items()):
                cols = st.columns([1, 3, 1])
                cols[0].write(f"**{name}**")
                # preview thumbnail
                mask = meta.get("mask")
                if mask is not None:
                    thumb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).copy()
                    overlay = thumb.copy()
                    # colorize mask
                    overlay[mask > 0] = (255, 0, 0)
                    alpha = 0.4
                    cv2.addWeighted(overlay, alpha, thumb, 1 - alpha, 0, thumb)
                    cols[1].image(thumb, use_column_width=True)
                cols[2].button("Delete", key=f"del_{name}", on_click=lambda n=name: (st.session_state["rois"].pop(n, None), st.session_state["inspector"].rois.pop(n, None), st.experimental_rerun()))
        else:
            st.info("No saved ROIs yet. Draw on the canvas or add by coordinates.")
        # Export/Import ROIs
        st.subheader("ROI Import/Export")
        col1, col2 = st.columns(2)
        # Export: convert masks to polygon contours for compact JSON
        def build_roi_export():
            export = {"width": w, "height": h, "rois": {}}
            for nm, meta in st.session_state["rois"].items():
                mask = meta.get("mask")
                tol = float(meta.get("tol", 0.0))
                contours, _ = cv2.findContours(mask.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                polys = []
                for c in contours:
                    pts = c.reshape(-1, 2).tolist()
                    polys.append(pts)
                export["rois"][nm] = {"tol": tol, "polys": polys}
            return export

        if st.button("Export ROIs as JSON"):
            export = build_roi_export()
            st.download_button("Download ROI JSON", data=json.dumps(export), file_name="rois.json", mime="application/json")

        uploaded = st.file_uploader("Import ROI JSON", type=["json"])
        if uploaded is not None:
            try:
                data = json.load(uploaded)
                # validate size
                if data.get("width") != w or data.get("height") != h:
                    st.warning("Imported ROI image size doesn't match current reference; adjust or re-upload matching reference.")
                else:
                    for nm, info in data.get("rois", {}).items():
                        mask = np.zeros((h, w), dtype=np.uint8)
                        polys = info.get("polys", [])
                        for poly in polys:
                            if poly:
                                pts = np.array(poly, dtype=np.int32)
                                cv2.fillPoly(mask, [pts], 255)
                        tol = float(info.get("tol", 0.02))
                        st.session_state["rois"][nm] = {"mask": mask, "tol": tol}
                        st.session_state["inspector"].add_roi(nm, mask, tol)
                    st.success("Imported ROIs")
                    st.experimental_rerun()
            except Exception as e:
                st.error(f"Failed to import ROI JSON: {e}")

        st.subheader("Inspect Single Part")
        test_file = st.file_uploader("Upload Test Part", type=["png", "jpg"]) 
        if test_file is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            tfile.write(test_file.read())
            tfile.close()

            metrics, ann_img = st.session_state["inspector"].inspect(tfile.name)
            try:
                inspections_total.inc()
                # optionally time inspection
            except Exception:
                pass

            col1, col2 = st.columns(2)
            col1.image(test_file, caption="Original Input")
            if ann_img is not None:
                col2.image(cv2.cvtColor(ann_img, cv2.COLOR_BGR2RGB), caption=f"Result: {metrics.get('status', 'N/A')}")

            st.json(metrics)
            # persist to DB
            try:
                db: DBEngine = st.session_state["db"]
                db.log_inspection(metrics, metrics.get("defects", []), metrics.get("part_id"))
            except Exception:
                pass
            os.unlink(tfile.name)
            
    with tab2:
        st.header("Batch Production Run")
        batch_files = st.file_uploader("Upload Multiple Parts for Run", type=["png", "jpg"], accept_multiple_files=True)
        if st.button("Run Batch Inspection") and batch_files:
            results = []
            progress = st.progress(0)
            for i, bf in enumerate(batch_files):
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                tfile.write(bf.read())
                tfile.close()

                metrics, _ = st.session_state["inspector"].inspect(tfile.name)
                try:
                    inspections_total.inc()
                    if metrics.get('status') == 'FAIL':
                        inspections_failed.inc()
                except Exception:
                    pass
                metrics["filename"] = bf.name
                results.append(metrics)
                # log to DB
                try:
                    st.session_state["db"].log_inspection(metrics, metrics.get("defects", []), metrics.get("part_id"))
                except Exception:
                    pass
                os.unlink(tfile.name)
                progress.progress((i + 1) / len(batch_files))

            st.session_state["batch_results"].extend(results)
            st.success(f"Processed {len(batch_files)} parts.")
            st.dataframe(results)
            
    with tab3:
        st.header("Statistical Process Control (SPC)")
        if len(st.session_state["batch_results"]) > 0:
            spc = SPCEngine(st.session_state["batch_results"])            
            col1, col2 = st.columns(2)
            # Area Capability
            target_area = st.session_state["inspector"].ref_outer_area
            tol_area = target_area * (size_tol / 100.0)
            cp, cpk, pp, ppk = spc.calculate_capability("outer_area", target_area, tol_area)
            col1.metric("Area Cp (Process Potential)", f"{cp:.2f}")
            col2.metric("Area Cpk (Process Capability)", f"{cpk:.2f}")

            # Alert if Cpk < 1.33
            if cpk < 1.33:
                st.warning(f"Process capability low: Cpk={cpk:.2f} < 1.33 — investigate!")

            # X-Bar chart (outer_area over samples)
            vals = [r.get("outer_area", 0.0) for r in st.session_state["batch_results"]]
            fig = px.line(y=vals, labels={"y": "Outer Area (px)"}, title="Outer Area over Samples")
            st.plotly_chart(fig, use_container_width=True)

            # Pareto: defect type counts
            defects_list = []
            for r in st.session_state["batch_results"]:
                for d in r.get("defects", []):
                    defects_list.append(d.get("type", "unknown"))
            if defects_list:
                df = px.data.tips()  # dummy to get px imports working
                # build counts
                from collections import Counter
                cnt = Counter(defects_list)
                pareto = px.bar(x=list(cnt.keys()), y=list(cnt.values()), title="Defect Pareto")
                st.plotly_chart(pareto, use_container_width=True)

            if st.button("Clear SPC History"):
                st.session_state["batch_results"] = []
                st.rerun()
        else:
            st.info("Run a batch inspection to generate SPC data.")
else:
    st.info("Please calibrate the system with a Golden Reference first.")
