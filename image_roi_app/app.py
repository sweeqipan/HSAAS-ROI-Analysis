import os
import cv2
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, send_file, session, abort
from werkzeug.utils import secure_filename
import io
import base64
import logging
import nibabel as nib
import time
import flask
import json # Keep for request parsing if needed elsewhere

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
app_start_time = time.strftime("%Y-%m-%d %H:%M:%S %Z")
logging.info(f"Application starting at: {app_start_time}")

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tif', 'tiff', 'nii', 'nii.gz'}
MAX_CONTENT_LENGTH = 128 * 1024 * 1024
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH
app.config['SECRET_KEY'] = os.urandom(24)

try:
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    logging.info(f"Upload folder '{UPLOAD_FOLDER}' ensured.")
except OSError as e:
    logging.error(f"CRITICAL: Cannot create upload folder '{UPLOAD_FOLDER}'. Error: {e}", exc_info=True)

# --- Helper Functions ---

def allowed_file(filename):
    """Checks file extension, handles .nii.gz."""
    if '.' not in filename: return False
    filename_lower = filename.lower()
    if filename_lower.endswith('.nii.gz'): return 'nii.gz' in ALLOWED_EXTENSIONS
    if '.' in filename_lower: simple_extension = filename_lower.rsplit('.', 1)[-1]; return simple_extension in ALLOWED_EXTENSIONS
    return False

def get_nifti_slice(nifti_img, slice_index=None, slice_dim_index=2):
    """Extracts, orients, and normalizes a specific or middle NIfTI slice."""
    # Keep the robust version from the previous complete code response
    try:
        logging.info(f"get_nifti_slice: req_idx={slice_index}, dim_idx={slice_dim_index}")
        img_data = nifti_img.get_fdata(dtype=np.float32, caching='unchanged')
        logging.info(f"NIfTI data shape: {img_data.shape}, dtype: {img_data.dtype}")
        img_ndim = img_data.ndim
        if not (0 <= slice_dim_index < img_ndim):
            slice_dim_index = min(2, img_ndim -1) if img_ndim >=3 else (1 if img_ndim >= 2 else 0); logging.warning(f"Using fallback slice dim: {slice_dim_index}")
        num_slices_in_dim = img_data.shape[slice_dim_index]
        logging.info(f"Slices in dim {slice_dim_index}: {num_slices_in_dim}")
        if slice_index is None: slice_to_extract = num_slices_in_dim // 2
        elif 0 <= slice_index < num_slices_in_dim: slice_to_extract = slice_index
        else: slice_to_extract = num_slices_in_dim // 2; logging.warning(f"Invalid slice index {slice_index}, using middle {slice_to_extract}")
        slice_to_extract = max(0, min(slice_to_extract, num_slices_in_dim - 1))
        logging.info(f"Will extract slice index {slice_to_extract} from dim {slice_dim_index}.")
        slice_obj = [slice(None)] * img_ndim
        slice_obj[slice_dim_index] = slice_to_extract
        spatial_dims_to_keep = {0, 1}
        for dim in range(img_ndim):
            if dim not in spatial_dims_to_keep and dim != slice_dim_index and img_data.shape[dim] > 1:
                if slice_obj[dim] == slice(None): slice_obj[dim] = 0; logging.info(f"Selecting index 0 from higher dim {dim}")
        logging.info(f"Constructed slice object: {slice_obj}")
        slice_data = img_data[tuple(slice_obj)]; logging.info(f"Slice extracted shape: {slice_data.shape}")
        if slice_data.ndim != 2:
             original_shape = slice_data.shape; slice_data = np.squeeze(slice_data); logging.warning(f"Attempted squeeze, new shape: {slice_data.shape}")
             if slice_data.ndim != 2: raise ValueError(f"Could not produce 2D slice (orig: {original_shape}, final: {slice_data.shape}).")
        slice_data = np.ascontiguousarray(slice_data, dtype=np.float32)
        slice_data_oriented = np.rot90(slice_data); logging.info("Applied np.rot90.")
        slice_display_norm = np.nan_to_num(slice_data_oriented).astype(np.float32)
        min_val, max_val = np.min(slice_display_norm), np.max(slice_display_norm)
        logging.info(f"Slice range: Min={min_val:.2f}, Max={max_val:.2f}")
        if max_val - min_val > 1e-6: normalized_slice = cv2.normalize(slice_display_norm, None, 0, 255, cv2.NORM_MINMAX)
        else: normalized_slice = np.full(slice_display_norm.shape, 128 if np.abs(min_val) > 1e-6 else 0, dtype=np.uint8)
        normalized_slice_uint8 = normalized_slice.astype(np.uint8)
        display_img_bgr = cv2.cvtColor(normalized_slice_uint8, cv2.COLOR_GRAY2BGR)
        logging.info(f"Display image created: shape={display_img_bgr.shape}, dtype={display_img_bgr.dtype}")
        return display_img_bgr, slice_data_oriented
    except Exception as e: logging.error(f"Error in get_nifti_slice: {e}", exc_info=True); raise ValueError(f"NIfTI slice error: {str(e)}") from e

# hex_to_bgr
def hex_to_bgr(hex_color):
    hex_color = hex_color.lstrip('#');
    if len(hex_color) != 6: return (0, 0, 0)
    try: rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4)); return (rgb[2], rgb[1], rgb[0])
    except ValueError: return (0, 0, 0)

# apply_threshold_colormap
def apply_threshold_colormap(slice_data, thresholds, colors_hex, background_color_hex="#000000"):
    logging.info(f"Applying colormap. #Thresholds: {len(thresholds)}")
    if slice_data is None or len(thresholds) != len(colors_hex): return np.zeros((*slice_data.shape, 3), dtype=np.uint8) if slice_data is not None else None
    slice_data_float = slice_data.astype(np.float32)
    output_image = np.zeros((*slice_data.shape, 3), dtype=np.uint8); output_image[:] = hex_to_bgr(background_color_hex)
    try:
        valid_thresholds = [{'value': float(t), 'color': c} for t, c in zip(thresholds, colors_hex)]; valid_thresholds.sort(key=lambda x: x['value'])
        sorted_thresholds_val = [t['value'] for t in valid_thresholds]; sorted_colors_hex = [t['color'] for t in valid_thresholds]
    except (ValueError, TypeError) as e: logging.error(f"Invalid threshold value: {e}"); return np.zeros((*slice_data.shape, 3), dtype=np.uint8)
    logging.debug(f"Sorted Thresh Vals: {sorted_thresholds_val}")
    for i in range(len(sorted_thresholds_val)):
        t_lower = sorted_thresholds_val[i]; t_upper = sorted_thresholds_val[i+1] if (i + 1) < len(sorted_thresholds_val) else np.inf; color_bgr = hex_to_bgr(sorted_colors_hex[i])
        mask = (slice_data_float >= t_lower) & (slice_data_float < t_upper); output_image[mask] = color_bgr; logging.debug(f"Applied {t_lower}<=val<{t_upper} with {color_bgr}")
    return output_image

# generate_threshold_mask - REMOVED as no longer needed

# calculate_roi_means
def calculate_roi_means(image_path_or_data, roi_x, roi_y, roi_size, grid_size, is_nifti=False, center_on_min=False, pix_dims=(1.0, 1.0), slice_index=None):
    # Keep the version with thinner green box & adjusted font
    initial_roi_x_req, initial_roi_y_req = roi_x, roi_y
    try:
        logging.info(f"Calculating ROI: req=({roi_x},{roi_y}), size={roi_size}, grid={grid_size}x{grid_size}, center={center_on_min}, nifti={is_nifti}, slice={slice_index}")
        img_display = None; img_calc_data = None; img_h = 0; img_w = 0
        if is_nifti:
            if not isinstance(image_path_or_data, np.ndarray): raise TypeError("Internal: Expected NumPy array for NIfTI.")
            img_calc_data = image_path_or_data.astype(np.float32); img_h, img_w = img_calc_data.shape[:2]; logging.info(f"Using NIfTI slice data: {img_calc_data.shape}")
            display_norm = np.nan_to_num(img_calc_data); min_val, max_val = np.min(display_norm), np.max(display_norm)
            if max_val - min_val > 1e-6: normalized_slice = cv2.normalize(display_norm, None, 0, 255, cv2.NORM_MINMAX)
            else: normalized_slice = np.full(display_norm.shape, 128 if np.abs(min_val) > 1e-6 else 0)
            img_display = cv2.cvtColor(normalized_slice.astype(np.uint8), cv2.COLOR_GRAY2BGR).copy()
        else:
            img = cv2.imread(image_path_or_data);
            if img is None: raise ValueError(f"Internal: Could not re-read: {image_path_or_data}")
            img_display = img.copy(); img_calc_data = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32); img_h, img_w = img_calc_data.shape[:2]; logging.info(f"Using std image: {img_calc_data.shape}")
        roi_x=max(0,int(roi_x)); roi_y=max(0,int(roi_y)); roi_size=max(10,int(roi_size)); roi_x=min(roi_x, img_w-roi_size); roi_y=min(roi_y, img_h-roi_size); roi_x=max(0,roi_x); roi_y=max(0,roi_y); roi_size=min(roi_size,img_w-roi_x,img_h-roi_y)
        if roi_size<=0: return {"error":"ROI invalid after clipping."},None,None,roi_x,roi_y
        initial_clipped_roi_x, initial_clipped_roi_y = roi_x, roi_y
        final_roi_x, final_roi_y = initial_clipped_roi_x, initial_clipped_roi_y
        if center_on_min:
            logging.info("Centering on min."); y_s,y_e=int(initial_clipped_roi_y),int(initial_clipped_roi_y+roi_size); x_s,x_e=int(initial_clipped_roi_x),int(initial_clipped_roi_x+roi_size)
            current_roi_data=img_calc_data[y_s:y_e, x_s:x_e]
            if current_roi_data.size > 0:
                min_val=np.min(current_roi_data); min_coords_rel=np.unravel_index(np.argmin(current_roi_data), current_roi_data.shape); min_y_rel, min_x_rel = min_coords_rel[0], min_coords_rel[1]; logging.info(f"Min val {min_val:.4f} at rel ({min_x_rel},{min_y_rel})")
                abs_min_x=initial_clipped_roi_x+min_x_rel; abs_min_y=initial_clipped_roi_y+min_y_rel; new_roi_x=round(abs_min_x-roi_size/2); new_roi_y=round(abs_min_y-roi_size/2); new_roi_x=max(0,int(new_roi_x)); new_roi_y=max(0,int(new_roi_y)); new_roi_x=min(new_roi_x, img_w-roi_size); new_roi_y=min(new_roi_y, img_h-roi_size); new_roi_x=max(0,new_roi_x); new_roi_y=max(0,new_roi_y)
                final_roi_x, final_roi_y = new_roi_x, new_roi_y; logging.info(f"ROI centered. New top-left: ({final_roi_x},{final_roi_y})")
            else: logging.warning("Cannot center: Initial ROI empty.")
        y_start,y_end=int(final_roi_y),int(final_roi_y+roi_size); x_start,x_end=int(final_roi_x),int(final_roi_x+roi_size)
        final_roi_calc_data = img_calc_data[y_start:y_end, x_start:x_end]
        if final_roi_calc_data.size == 0: return {"error":"Final ROI empty."},None,None,final_roi_x,final_roi_y
        main_mean = np.mean(final_roi_calc_data); results = {'means': [], 'main_mean': f"{main_mean:.3f}", 'roi_coords': {'x': final_roi_x, 'y': final_roi_y, 'size': roi_size}}; logging.info(f"Main ROI Mean: {main_mean:.4f}")
        roi_data_for_excel = []; pix_dim_x, pix_dim_y = pix_dims; roi_width_mm=roi_size*pix_dim_x; roi_height_mm=roi_size*pix_dim_y; main_area_mm2=roi_width_mm*roi_height_mm
        excel_main_row = {'ROI_Type': 'Main', 'Grid_Pos': 'N/A', 'X (px)': final_roi_x, 'Y (px)': final_roi_y, 'Size (px)': roi_size, 'Width (mm)': roi_width_mm, 'Height (mm)': roi_height_mm, 'Area (mm^2)': main_area_mm2, 'Mean_Value': main_mean}
        if is_nifti and slice_index is not None: excel_main_row['Slice_Index'] = slice_index
        roi_data_for_excel.append(excel_main_row)
        cv2.rectangle(img_display, (final_roi_x, final_roi_y), (final_roi_x + roi_size, final_roi_y + roi_size), (0, 255, 0), 1) # Thickness 1
        grid_size = int(grid_size)
        if grid_size > 1:
            sub_roi_size_px = roi_size // grid_size
            if sub_roi_size_px > 0:
                 for i in range(grid_size):
                    for j in range(grid_size):
                        sub_x_px=final_roi_x+j*sub_roi_size_px; sub_y_px=final_roi_y+i*sub_roi_size_px; sub_y_start_rel,sub_y_end_rel=i*sub_roi_size_px,(i+1)*sub_roi_size_px; sub_x_start_rel,sub_x_end_rel=j*sub_roi_size_px,(j+1)*sub_roi_size_px
                        sub_roi_calc_data = final_roi_calc_data[sub_y_start_rel:sub_y_end_rel, sub_x_start_rel:sub_x_end_rel]
                        sub_mean_val = np.nan; mean_str = "N/A"
                        if sub_roi_calc_data.size > 0:
                            sub_mean_val = np.mean(sub_roi_calc_data); mean_str = f"{sub_mean_val:.2f}"
                            cv2.rectangle(img_display, (sub_x_px, sub_y_px), (sub_x_px + sub_roi_size_px, sub_y_px + sub_roi_size_px), (255, 0, 0), 1)
                            MIN_SIZE_FOR_TEXT = 12
                            if sub_roi_size_px > MIN_SIZE_FOR_TEXT: text_x = sub_x_px + 2; text_y = sub_y_px + 8; font_scale = 0.22 if grid_size >= 4 else 0.28; font_thickness = 1; cv2.putText(img_display, mean_str, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 0), font_thickness)
                        results['means'].append(mean_str); sub_width_mm=sub_roi_size_px*pix_dim_x; sub_height_mm=sub_roi_size_px*pix_dim_y; sub_area_mm2=sub_width_mm*sub_height_mm
                        excel_sub_row = {'ROI_Type': 'Sub', 'Grid_Pos': f'{i+1}x{j+1}', 'X (px)': sub_x_px, 'Y (px)': sub_y_px, 'Size (px)': sub_roi_size_px, 'Width (mm)': sub_width_mm, 'Height (mm)': sub_height_mm, 'Area (mm^2)': sub_area_mm2, 'Mean_Value': sub_mean_val}
                        if is_nifti and slice_index is not None: excel_sub_row['Slice_Index'] = slice_index
                        roi_data_for_excel.append(excel_sub_row)
            else: logging.warning(f"Sub-ROI size zero.")
        logging.info("Finished grid calculations.")
        success, buffer = cv2.imencode('.png', img_display); # Encode PNG
        if not success: raise ValueError("Failed to encode PNG.")
        img_base64 = base64.b64encode(buffer).decode('utf-8'); logging.info("Encoded PNG.")
        return results, img_base64, roi_data_for_excel, final_roi_x, final_roi_y
    except Exception as e: logging.error(f"Error in calculate_roi_means: {e}", exc_info=True); return {"error": f"Calc error: {str(e)}"}, None, None, initial_roi_x_req, initial_roi_y_req


# --- Flask Routes ---

@app.route('/')
def index():
    logging.info("Serving index page, clearing session."); session.clear(); return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    """Handle image upload, return slice info for NIfTI."""
    # Use robust version, ensure absolute path stored
    logging.info("=== Received upload request ===")
    if 'file' not in request.files: return jsonify({"error": "No file part"}), 400
    file = request.files['file']; original_filename = file.filename
    if original_filename == '': return jsonify({"error": "No file selected"}), 400
    logging.info(f"Upload requested for: '{original_filename}'")
    if not allowed_file(original_filename): return jsonify({"error": f"File type not allowed. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"}), 400
    filepath = None
    try:
        filename = secure_filename(original_filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        logging.info(f"Saving to: '{filepath}'"); file.save(filepath); logging.info("File saved.")
        absolute_filepath = os.path.abspath(filepath) # *** STORE ABSOLUTE PATH ***
        if not os.path.exists(absolute_filepath): raise IOError("File saving failed silently.")
        logging.info(f"Confirmed exists: '{absolute_filepath}'. Size: {os.path.getsize(absolute_filepath)} bytes.")
        is_nifti = filename.lower().endswith(('.nii', '.nii.gz'))
        pix_dims = [1.0, 1.0]; img_width_px, img_height_px = 0, 0; display_image_data_url = None; num_slices = 1; slice_dim_index = -1; middle_slice_index = 0
        if is_nifti:
            session['is_nifti'] = True; session['nifti_path'] = absolute_filepath # Store absolute path
            logging.info(f"Processing NIfTI: '{absolute_filepath}'")
            nifti_img = nib.load(absolute_filepath); logging.info(f"NIfTI loaded. Shape={nifti_img.shape}")
            img_ndim = nifti_img.ndim
            if img_ndim >= 3 and nifti_img.shape[2] > 1: slice_dim_index = 2
            elif img_ndim >= 2 and nifti_img.shape[1] > 1: slice_dim_index = 1
            elif img_ndim >= 1 and nifti_img.shape[0] > 1: slice_dim_index = 0
            else: slice_dim_index = -1
            if slice_dim_index != -1: num_slices = nifti_img.shape[slice_dim_index]; middle_slice_index = max(0, min(num_slices // 2, num_slices - 1))
            else: num_slices = 1; middle_slice_index = 0; logging.warning(f"Cannot determine slice dim for {nifti_img.shape}.")
            session['num_slices'] = num_slices; session['slice_dim_index'] = slice_dim_index
            try:
                 zooms = nifti_img.header.get_zooms(); pix_dims = [abs(float(zooms[0])), abs(float(zooms[1]))]
                 if pix_dims[0] < 1e-9: pix_dims[0] = 1.0
                 if pix_dims[1] < 1e-9: pix_dims[1] = 1.0; logging.info(f"NIfTI pix dims: {pix_dims}")
            except Exception: pix_dims = [1.0, 1.0]; logging.warning("Using default pix dims.")
            display_img_np, _ = get_nifti_slice(nifti_img, middle_slice_index, slice_dim_index)
            img_height_px, img_width_px = display_img_np.shape[:2]
            success, buffer = cv2.imencode('.png', display_img_np) # Encode PNG
            if not success: raise ValueError("PNG encode failed.")
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            display_image_data_url = f"data:image/png;base64,{img_base64}"; logging.info("Encoded middle NIfTI slice as PNG Base64.")
        else:
            session['is_nifti'] = False; session['image_path'] = absolute_filepath # Store absolute path
            session['num_slices'] = 1; session['slice_dim_index'] = -1
            logging.info(f"Processing standard image: '{absolute_filepath}'")
            img = cv2.imread(absolute_filepath);
            if img is None: raise ValueError(f"Could not read image file '{filename}'.")
            img_height_px, img_width_px = img.shape[:2]; pix_dims = [1.0, 1.0]
            display_image_data_url = f'/uploads/{filename}'; logging.info(f"Std image loaded: {img_width_px}x{img_height_px}")
        session['filename'] = filename; session['pix_dim_x'] = pix_dims[0]; session['pix_dim_y'] = pix_dims[1]
        session['image_width_px'] = img_width_px; session['image_height_px'] = img_height_px
        logging.info("Session updated.")
        return jsonify({ "message": "File uploaded successfully", "image_url": display_image_data_url, "image_width": img_width_px, "image_height": img_height_px, "filename": filename, "pix_dim_x": pix_dims[0], "pix_dim_y": pix_dims[1], "is_nifti": is_nifti, "num_slices": num_slices, "slice_dim_index": slice_dim_index, "middle_slice_index": middle_slice_index })
    except Exception as e:
         logging.error(f"--- Upload FAILED for '{original_filename}' --- Error: {e}", exc_info=True)
         if filepath and os.path.exists(filepath):
             try: os.remove(filepath); logging.info(f"Removed '{filepath}'.")
             except OSError as rm_error: logging.error(f"Could not remove '{filepath}': {rm_error}")
         session.clear()
         user_error_message = str(e) if isinstance(e, (ValueError, IOError, nib.filebasedimages.ImageFileError)) else "Server error during upload."
         logging.error(f"Returning error to client: {user_error_message}")
         return jsonify({"error": user_error_message}), 500

@app.route('/get_slice/<int:slice_index>')
def get_slice_data_route(slice_index):
    """Fetches and returns a specific NIfTI slice as a base64 encoded PNG image."""
    # Use version with corrected path checking / more logging
    logging.info(f"Request for NIfTI slice index: {slice_index}")
    if not session.get('is_nifti'): logging.error("/get_slice: No NIfTI file in session."); return jsonify({"error": "No NIfTI file loaded."}), 400
    nifti_path = session.get('nifti_path'); num_slices = session.get('num_slices'); slice_dim_index = session.get('slice_dim_index')
    logging.info(f"/get_slice: Session nifti_path='{nifti_path}', num_slices={num_slices}, slice_dim_index={slice_dim_index}")
    if not nifti_path: logging.error("/get_slice: nifti_path missing from session."); session.clear(); return jsonify({"error": "NIfTI path missing. Re-upload."}), 404
    if not os.path.exists(nifti_path): logging.error(f"/get_slice: File path does not exist: '{nifti_path}'"); session.clear(); return jsonify({"error": "NIfTI file not found. Re-upload."}), 404
    if num_slices is None or slice_dim_index is None or slice_dim_index < 0: logging.error("/get_slice: Invalid slice info in session."); return jsonify({"error": "Slice info invalid."}), 400
    if not (0 <= slice_index < num_slices): logging.error(f"/get_slice: Invalid slice index {slice_index} for num_slices {num_slices}."); return jsonify({"error": f"Invalid slice index {slice_index}."}), 400
    try:
        logging.info(f"Loading NIfTI for slice {slice_index}: {nifti_path}")
        nifti_img = nib.load(nifti_path)
        display_img_np, _ = get_nifti_slice(nifti_img, slice_index, slice_dim_index); logging.info(f"Slice {slice_index} extracted.")
        success, buffer = cv2.imencode('.png', display_img_np);
        if not success: raise ValueError("cv2.imencode (PNG) failed for slice.")
        img_base64 = base64.b64encode(buffer).decode('utf-8'); logging.info(f"Slice {slice_index} encoded successfully (PNG).")
        return jsonify({ "message": f"Slice {slice_index} processed.", "image_url": f"data:image/png;base64,{img_base64}", "slice_index": slice_index })
    except Exception as e:
        logging.error(f"Error getting slice {slice_index} for '{nifti_path}': {e}", exc_info=True)
        user_error_message = str(e) if isinstance(e, (ValueError, IOError, nib.filebasedimages.ImageFileError)) else f"Server error processing slice {slice_index}."
        return jsonify({"error": user_error_message}), 500


@app.route('/get_slice_info/<int:slice_index>')
def get_slice_info(slice_index):
    """Gets min/max info for a specific NIfTI slice."""
    # Keep robust version
    logging.info(f"Request slice info index: {slice_index}")
    if not session.get('is_nifti'): return jsonify({"error": "No NIfTI file loaded."}), 400
    nifti_path = session.get('nifti_path'); num_slices = session.get('num_slices'); slice_dim_index = session.get('slice_dim_index')
    if not nifti_path or not os.path.exists(nifti_path): session.clear(); return jsonify({"error": "NIfTI path invalid. Re-upload."}), 404
    if num_slices is None or slice_dim_index is None or slice_dim_index < 0: return jsonify({"error": "Slice info invalid."}), 400
    if not (0 <= slice_index < num_slices): return jsonify({"error": f"Invalid slice index {slice_index}."}), 400
    try:
        nifti_img = nib.load(nifti_path); _, slice_data_oriented = get_nifti_slice(nifti_img, slice_index, slice_dim_index)
        min_val = np.min(slice_data_oriented); max_val = np.max(slice_data_oriented); logging.info(f"Slice {slice_index} Min: {min_val}, Max: {max_val}")
        return jsonify({ "slice_index": slice_index, "min_value": float(min_val) if not np.isnan(min_val) else None, "max_value": float(max_val) if not np.isnan(max_val) else None })
    except Exception as e: logging.error(f"Error getting slice info {slice_index}: {e}", exc_info=True); return jsonify({"error": f"Server error: {str(e)}"}), 500


@app.route('/get_thresholded_slice', methods=['POST'])
def get_thresholded_slice():
    """Applies thresholds/colors and returns the result as base64 PNG."""
    logging.info("Request thresholded slice.")
    if not session.get('is_nifti'):
        return jsonify({"error": "No NIfTI file loaded."}), 400

    nifti_path = session.get('nifti_path')
    slice_dim_index = session.get('slice_dim_index') # Get slice dimension

    if not nifti_path or not os.path.exists(nifti_path):
        session.clear()
        return jsonify({"error": "NIfTI path invalid. Re-upload."}), 404

    # *** CORRECTED PART BELOW ***
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON data received."}), 400 # Check after getting data

    slice_index = data.get('slice_index')
    thresholds_data = data.get('thresholds') # Get thresholds from JSON body
    # *** END OF CORRECTION ***

    if slice_index is None or thresholds_data is None:
        return jsonify({"error": "Missing slice_index or thresholds in request."}), 400

    num_slices = session.get('num_slices', 0)
    if not (0 <= slice_index < num_slices):
        return jsonify({"error": "Invalid slice index."}), 400

    try:
        # Extract and sort threshold values and colors
        valid_thresholds = []
        for t in thresholds_data:
            try:
                # Ensure value exists and can be converted to float
                value = t.get('value')
                if value is None: continue # Skip if no value key
                val = float(value)
                col = str(t.get('color', '#000000')) # Default to black if color missing
                if not col.startswith('#') or len(col) not in [4, 7]: col = '#000000' # Basic hex validation
                valid_thresholds.append({'value': val, 'color': col})
            except (ValueError, TypeError, AttributeError):
                logging.warning(f"Skipping invalid threshold entry: {t}")

        if not valid_thresholds:
            return jsonify({"error": "No valid thresholds provided."}), 400

        valid_thresholds.sort(key=lambda x: x['value'])
        threshold_values = [t['value'] for t in valid_thresholds]
        color_values_hex = [t['color'] for t in valid_thresholds]

        # Load image and get slice data
        nifti_img = nib.load(nifti_path)
        _, slice_data_oriented = get_nifti_slice(nifti_img, slice_index, slice_dim_index)

        # Apply colormap
        colored_image = apply_threshold_colormap(slice_data_oriented, threshold_values, color_values_hex)
        if colored_image is None: raise ValueError("Colormap application failed.")

        # Encode result as PNG
        success, buffer = cv2.imencode('.png', colored_image)
        if not success: raise ValueError("Failed to encode thresholded image.")
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        logging.info(f"Thresholded slice {slice_index} generated and encoded.")
        return jsonify({
            "message": "Thresholded slice generated.",
            "image_url": f"data:image/png;base64,{img_base64}", # Use PNG prefix
            "slice_index": slice_index
        })

    except Exception as e:
        logging.error(f"Error generating thresholded slice {slice_index}: {e}", exc_info=True)
        user_error_message = str(e) if isinstance(e, (ValueError, TypeError)) else "Server error generating thresholded image."
        return jsonify({"error": user_error_message}), 500

@app.route('/download_colormap_png', methods=['POST'])
def download_colormap_png():
    """Generates and sends the color mapped slice as a PNG file."""
    logging.info("Request colormap PNG download.")
    if not session.get('is_nifti'):
        abort(400, "No NIfTI file loaded.") # Use abort correctly

    nifti_path = session.get('nifti_path')
    slice_dim_index = session.get('slice_dim_index') # Get slice dimension

    if not nifti_path or not os.path.exists(nifti_path):
        abort(404, "NIfTI file not found.") # Use abort correctly

    # *** CORRECTED PART BELOW ***
    data = request.get_json()
    if not data:
        abort(400, "No JSON data received.") # Check on separate line

    slice_index = data.get('slice_index')
    thresholds_data = data.get('thresholds')
    # *** END OF CORRECTION ***

    if slice_index is None or thresholds_data is None:
        abort(400, "Missing slice_index or thresholds.") # Use abort correctly

    num_slices = session.get('num_slices', 0)
    if not (0 <= slice_index < num_slices):
        abort(400, "Invalid slice index.") # Use abort correctly

    try:
        # Extract and sort threshold values and colors
        valid_thresholds = []
        for t in thresholds_data:
             try:
                 # Ensure value exists and can be converted to float
                 value = t.get('value')
                 if value is None: continue
                 val = float(value)
                 col = str(t.get('color', '#000000'))
                 if not col.startswith('#') or len(col) not in [4, 7]: col = '#000000'
                 valid_thresholds.append({'value': val, 'color': col})
             except (ValueError, TypeError, AttributeError):
                 logging.warning(f"Skipping invalid threshold for download: {t}")

        if not valid_thresholds:
            abort(400, "No valid thresholds provided.") # Use abort correctly

        valid_thresholds.sort(key=lambda x: x['value'])
        threshold_values = [t['value'] for t in valid_thresholds]
        color_values_hex = [t['color'] for t in valid_thresholds]

        # Load NIfTI, get slice, apply colormap
        nifti_img = nib.load(nifti_path)
        _, slice_data_oriented = get_nifti_slice(nifti_img, slice_index, slice_dim_index)
        colored_image = apply_threshold_colormap(slice_data_oriented, threshold_values, color_values_hex)
        if colored_image is None: raise ValueError("Colormap application failed.")

        # Encode as PNG
        success, buffer = cv2.imencode('.png', colored_image)
        if not success: raise ValueError("PNG encoding failed.")

        # Prepare buffer for sending
        img_io = io.BytesIO(buffer)
        img_io.seek(0)

        # Prepare filename
        base_filename, _ = os.path.splitext(session.get('filename', 'colormap'))
        download_filename = f"{secure_filename(base_filename)}_slice{slice_index}_colormap.png"
        logging.info(f"Sending colormap PNG: {download_filename}")

        # Send file
        return send_file(img_io, mimetype='image/png', as_attachment=True, download_name=download_filename)

    except Exception as e:
        logging.error(f"Error generating colormap PNG for slice {slice_index}: {e}", exc_info=True)
        # Return plain text error for download routes
        return f"Error generating colormap PNG: {str(e)}", 500
# --- REMOVED /download_mask_nifti route ---

@app.route('/calculate_roi', methods=['POST'])
def calculate_roi():
    """Handles ROI calculation requests, converting numpy types before session save."""
    # Keep robust version with numpy type conversion fix
    logging.info("Received ROI calculation request.")
    data = request.get_json();
    if not data: return jsonify({"error": "No data received"}), 400
    logging.info(f"Calculation parameters received: {data}")
    filename = session.get('filename'); is_nifti = session.get('is_nifti', False)
    roi_x = data.get('x'); roi_y = data.get('y'); roi_size = data.get('size'); grid_size = data.get('grid', 1); center_on_min = data.get('center_on_min', False); requested_slice_index = data.get('slice_index', None)
    pix_dim_x = session.get('pix_dim_x', 1.0); pix_dim_y = session.get('pix_dim_y', 1.0)
    if not filename: return jsonify({"error": "No image in session. Re-upload."}), 400
    if roi_x is None or roi_y is None or roi_size is None: return jsonify({"error": "Missing ROI params"}), 400
    image_data_or_path = None; slice_index_to_use = None
    try:
        if is_nifti:
            nifti_path = session.get('nifti_path');
            if not nifti_path or not os.path.exists(nifti_path): raise FileNotFoundError("NIfTI path missing/invalid.")
            nifti_img = nib.load(nifti_path); num_slices = session.get('num_slices', 0); slice_dim_index = session.get('slice_dim_index', -1)
            if requested_slice_index is not None and 0 <= requested_slice_index < num_slices: slice_index_to_use = requested_slice_index
            else: slice_index_to_use = num_slices // 2; logging.warning(f"Using middle slice {slice_index_to_use} for calc.")
            slice_index_to_use = max(0, min(slice_index_to_use, num_slices -1))
            _, image_data_or_path = get_nifti_slice(nifti_img, slice_index_to_use, slice_dim_index); logging.info(f"Using NIfTI slice {slice_index_to_use} for calculation.")
        else:
            image_path = session.get('image_path');
            if not image_path or not os.path.exists(image_path): raise FileNotFoundError("Image path missing/invalid.")
            image_data_or_path = image_path; logging.info(f"Using std image path for calculation: {image_path}")
        results, img_base64, roi_data, final_x, final_y = calculate_roi_means(image_data_or_path, roi_x, roi_y, roi_size, grid_size, is_nifti=is_nifti, center_on_min=center_on_min, pix_dims=(pix_dim_x, pix_dim_y), slice_index=slice_index_to_use)
        if "error" in results: return jsonify({"error": results["error"], "final_roi_x": final_x, "final_roi_y": final_y}), 400
        if roi_data:
            serializable_roi_data = []
            for row_dict in roi_data:
                new_row = {};
                for key, value in row_dict.items():
                    if isinstance(value, np.integer): new_row[key] = int(value)
                    elif isinstance(value, np.floating): new_row[key] = float(value) if not np.isnan(value) else None
                    elif isinstance(value, np.bool_): new_row[key] = bool(value)
                    else: new_row[key] = value
                serializable_roi_data.append(new_row)
            session['roi_data'] = serializable_roi_data; logging.info("Stored serializable excel data in session.")
        else: session['roi_data'] = []; logging.warning("roi_data empty after calculation.")
        results['roi_coords']['x'] = final_x; results['roi_coords']['y'] = final_y
        return jsonify({ "results": results, "annotated_image": f"data:image/png;base64,{img_base64}", "final_roi_x": final_x, "final_roi_y": final_y }) # Ensure PNG prefix
    except FileNotFoundError as e: logging.error(f"File not found: {e}"); session.clear(); return jsonify({"error": f"Error: {str(e)}. Session cleared. Re-upload."}), 404
    except Exception as e: logging.error(f"Error during ROI calc req: {e}", exc_info=True); return jsonify({"error": f"Server error during calc: {str(e)}"}), 500

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serves uploaded standard image files only, with robust path checking."""
    # Keep robust version (with fixed syntax error)
    if session.get('is_nifti') and filename == session.get('filename'):
        logging.warning(f"Blocked attempt to serve NIfTI file directly: {filename}")
        abort(403, "Forbidden: Cannot serve NIfTI file directly.")
    logging.info(f"Request to serve standard uploaded file: {filename}")
    try:
        safe_dir = os.path.abspath(app.config['UPLOAD_FOLDER'])
        safe_filename = secure_filename(filename)
        # *** Check safe_filename BEFORE using it ***
        if not safe_filename:
            logging.error(f"Invalid filename requested (unsafe): {filename}")
            abort(400, "Invalid filename.")

        safe_path = os.path.join(safe_dir, safe_filename)
        logging.info(f"Serving resolved path: {safe_path}")

        if not os.path.exists(safe_path) or not os.path.isfile(safe_path) or os.path.commonpath([safe_path, safe_dir]) != safe_dir:
            logging.error(f"File not found or path invalid: {safe_path}")
            abort(404, "File not found or path is invalid.")

        return send_file(safe_path)
    except Exception as e: logging.error(f"Error serving file {filename}: {e}", exc_info=True); abort(500)


@app.route('/download_excel', methods=['GET'])
def download_excel():
    """Generates and sends the ROI results as an Excel file."""
    # Keep robust version
    logging.info("Request download Excel."); roi_data = session.get('roi_data'); original_filename = session.get('filename', 'roi_data')
    if not roi_data: return "No ROI data available.", 404
    try:
        df = pd.DataFrame(roi_data); logging.info(f"DF for Excel: {df.shape}")
        base_cols = ['ROI_Type', 'Grid_Pos', 'X (px)', 'Y (px)', 'Size (px)', 'Width (mm)', 'Height (mm)', 'Area (mm^2)', 'Mean_Value']
        column_order = base_cols.copy();
        if 'Slice_Index' in df.columns: column_order.insert(2, 'Slice_Index')
        else: df['Slice_Index'] = 'N/A'
        for col in column_order:
            if col not in df.columns: df[col] = np.nan
        df = df[column_order]
        for col in ['Width (mm)', 'Height (mm)', 'Area (mm^2)', 'Mean_Value']:
             if col in df.columns: df[col] = df[col].apply(lambda x: f"{x:.3f}" if pd.notna(x) and isinstance(x, (int, float, np.number)) else ('NaN' if pd.isna(x) else x))
        if 'Slice_Index' in df.columns: df['Slice_Index'] = df['Slice_Index'].apply(lambda x: int(x) if pd.notna(x) and isinstance(x, (int, float, np.number)) and x != 'N/A' else ('N/A' if x == 'N/A' else x))
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, sheet_name='ROI Means', index=False); worksheet = writer.sheets['ROI Means']
            for idx, col_name in enumerate(df.columns): series = df[col_name]; max_len = max((series.astype(str).map(len).max(), len(str(col_name)))) + 2; worksheet.set_column(idx, idx, max_len)
        output.seek(0); logging.info("Excel generated.")
        base_filename, _ = os.path.splitext(original_filename); excel_filename = f"{secure_filename(base_filename)}_roi_means.xlsx"
        return send_file( output, mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', as_attachment=True, download_name=excel_filename )
    except Exception as e: logging.error(f"Error generating Excel: {e}", exc_info=True); return f"Error generating Excel: {str(e)}", 500

# --- Main Execution Guard ---
if __name__ == '__main__':
    logging.info(f"--- Starting Flask Development Server (PID: {os.getpid()}) ---")
    try: from importlib import metadata; logging.info(f"Flask Version: {metadata.version('flask')}"); logging.info(f"OpenCV Version: {cv2.__version__}"); logging.info(f"Numpy Version: {np.__version__}"); logging.info(f"Pandas Version: {pd.__version__}"); logging.info(f"Nibabel Version: {nib.__version__}")
    except Exception as e: logging.warning(f"Could not log library versions: {e}")
    logging.info(f"Upload folder: {os.path.abspath(UPLOAD_FOLDER)}"); logging.info(f"Allowed Exts: {ALLOWED_EXTENSIONS}"); logging.info(f"Max Size: {app.config['MAX_CONTENT_LENGTH'] / (1024*1024):.1f} MB")
    logging.info(f"Server starting on http://0.0.0.0:5000"); app.run(debug=True, host='0.0.0.0', port=5000)