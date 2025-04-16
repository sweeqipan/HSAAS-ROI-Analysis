# HSAAS-ROI-Analysis

# Customized ROI Analysis Tool for HSAAS

**Caution: For Investigational Use Only.**

This tool allows users to load standard images (PNG, JPG, etc.) and NIfTI medical images (`.nii`, `.nii.gz`), perform Region of Interest (ROI) analysis (calculating mean values within single or gridded ROIs), and apply custom thresholding/color mapping (for NIfTI files). Results can be exported.

## Prerequisites

* **Windows Operating System** (Tested on Windows 10/11)
* **Python:** Version 3.8 or newer recommended.
* **Internet Connection:** Required for downloading Python (if needed) and installing libraries.

## Step 1: Check for / Install Python

You need Python installed to run the tool's backend server. Python also comes with `pip`, the package installer we'll use.

1.  **Check if Python is Installed:**
    * Open the **PowerShell**. You can find these by searching in the Windows Start menu.
    * Type the following command and press Enter:
        ```bash
        python --version
        ```
    * If you see a version number (e.g., `Python 3.11.4`), Python is installed. Note the version. If it's older than 3.8, consider upgrading.
    * If you get an error message like "'python' is not recognized..." or "'py' is not recognized...", Python is likely not installed or not added to your system's PATH.

2.  **Install Python (If Needed):**
    * Go to the official Python website: [https://www.python.org/downloads/](https://www.python.org/downloads/)
    * Download the latest stable **Windows installer** (usually ends in `.exe`).
    * Run the downloaded installer.
    * **IMPORTANT:** On the first screen of the installer, make sure to check the box at the bottom that says **"Add Python X.X to PATH"** (where X.X is the version number). This makes Python easily accessible from the command line.
    * Continue with the default installation options (click "Install Now"). Administrator permission may be required.
    * Once installation is complete, close and reopen the Command Prompt/PowerShell and try the `python --version` or `py --version` command again to verify.

## Step 2: Get the Tool's Code

You need to download the application code from the GitHub repository.


**Downloading ZIP File**

1.  Go to the main page of the GitHub repository in your web browser.
2.  Click the green **"< > Code"** button.
3.  Select **"Download ZIP"**.
4.  Save the ZIP file to a location you can easily find (e.g., your Downloads folder).
5.  Go to where you saved the ZIP file. Right-click on it and choose **"Extract All..."**.
6.  Choose a destination for the extracted files (e.g., create a folder on your Desktop called `image_roi_app`). Click **"Extract"**.
7.  Anywhere will do, but make sure you know the full path to this extracted `image_roi_app` folder.

## Step 3: Set Up Environment and Install Dependencies

This step installs the Python libraries the tool needs to function. Using a virtual environment is recommended to keep dependencies separate for different projects.

1.  **Open PowerShell.**
   
2.  **Navigate to the Project Directory:** Use the `cd` command to go into the `image_roi_app` folder you created in Step 2.
    * `cd C:\Users\YourUsername\Desktop\image_roi_app`
    * *(Adjust the path based on where you actually put the folder)*. You should see files like `app.py` if you type `ls` (in powershell).
      
3.  **Install Required Libraries:**
    * Run the following command:
        python -m pip install Flask opencv-python numpy pandas xlsxwriter nibabel waitress
    * This command uses `pip` (Python's package installer) to download and install all the necessary libraries specified in the code (`Flask` for the web server, `opencv-python` for image processing, `numpy` for numerical operations, `pandas` and `xlsxwriter` for Excel export, `nibabel` for NIfTI files, and `waitress` for a more robust server option). Wait for the installation to complete. You might see progress bars or status messages.

## Step 4: Run the Application

Now that the environment is set up, you can start the tool's web server.

1.  **Ensure you are in the correct directory** (`image_roi_app`) in your PowerShell.

2.  **Run the Python script:**
    python app.py
    
4.  **Observe the Output:** You should see several lines of output in the terminal, including:
    * Logging information (versions, folders).
    * `* Running on http://127.0.0.1:5000`
    * `* Running on http://YOUR_COMPUTER_IP:5000` (This address allows access from other devices on your local network).
    * `* Debug mode: on`
    * A Debugger PIN (you usually don't need this unless troubleshooting).
    * **Important:** Keep this terminal window open! Closing it will stop the application server.

## Step 5: Access the Tool in Your Browser

1.  Open your preferred web browser (Chrome, Firefox, Edge, etc.).
   
2.  In the address bar, type:
    [http://127.0.0.1:5000]
    or
    http://localhost:5000
    
3.  Press Enter. The "Customized ROI Analysis Tool for HSAAS" interface should load.

## Step 6: Using the Tool - Function Guide

1.  **Upload Image (Section 1):**
    * Click "Choose File".
    * Select a standard image (`.png`, `.jpg`, `.jpeg`, `.bmp`, `.tif`, `.tiff`) or a NIfTI image (`.nii`, `.nii.gz`).
    * Click the "Upload" button.
    * The image will appear in the left panel. For NIfTI files, the middle slice is shown initially.

2.  **NIfTI Slice Navigation (Left Panel - NIfTI Only):**
    * If a multi-slice NIfTI file was uploaded, a "Slice" slider will appear above the image.
    * Drag the slider or use the **mouse wheel** while hovering over the image display area to change the currently viewed slice. The display updates automatically.

3.  **ROI Controls (Section 2):**
    * **Enable ROI Box:** Check this box to activate ROI functions. A red square will appear on the image.
    * **Position ROI:** Click *inside* the red box and drag it to the desired location.
    * **Resize ROI:** Use the "ROI Size (Side)" slider to adjust the size (in pixels) of the square ROI. The corresponding physical size (if pixel dimensions are known from NIfTI) will update below the slider.
    * **Grid Division:** Select 1x1 (single ROI), 2x2, 3x3, 4x4, or 5x5. The red box overlay will show faint blue grid lines for divisions > 1x1.
    * **Center on Min:** (Optional) Click this button *after* placing the ROI. It will recalculate the position to center the box on the pixel with the lowest intensity *within the original box area* and then perform the mean calculation based on the *new* centered position.
    * **Recalculate Means:** Click this button after positioning/resizing the ROI or changing the grid to perform the mean value calculation(s). The results appear in Section 3, and the image display updates with a green ROI boundary and blue sub-grid boundaries/mean values overlaid (mean values only shown if sub-ROIs are large enough).

4.  **ROI Results (Section 3):**
    * Displays the calculated "Main ROI Mean".
    * Displays the individual mean values for each sub-ROI if a grid division (> 1x1) was selected.

5.  **Export ROI Results (Section 4):**
    * **Download Mean Values (Excel):** After clicking "Recalculate Means", this button becomes active. Click it to download an Excel (`.xlsx`) file containing the coordinates, size (pixels and mm), area (mm²), mean value for the main ROI and all sub-ROIs, and the slice index (for NIfTI).

6.  **Thresholding & Colormap (Section 5 - NIfTI Only):**
    * This section only appears if a NIfTI file is loaded.
    * **Enable Thresholding:** Check this box to activate the feature. The image display will switch from grayscale to black initially.
    * **Default Thresholds:** When first enabled, a default color map is applied:
        * `< 100`: Black (Background)
        * `100` to `399.99...`: Blue
        * `400` to `799.99...`: Yellow
        * `800` to `999.99...`: Green
        * `>= 1000`: Red
    * **Slice Range:** Shows the minimum and maximum intensity values found in the currently displayed slice.
    * **Threshold List:**
        * Each row defines a lower bound (Value ≥ Threshold) and the color applied to pixels in that range (up to the next threshold's value).
        * Modify the `Threshold ≥` number input to change the threshold value.
        * Click the color swatch (`<input type="color">`) to choose a different color for that range using a color picker.
        * The image display updates automatically (after a short delay) when you change values or colors.
    * **Add/Remove:** Use the `+` button to add a new threshold level below the existing ones. Use the `-` button on any row to remove that specific threshold level.
    * **Export Color Map (PNG):** Click this to download the currently displayed color-mapped image as a PNG file.

## Step 7: Stopping the Application

1.  Go back to the **PowerShell window** where the server is running (the one showing all the log messages).
2.  Press **Ctrl + C**.
3.  You might be asked "Terminate batch job (Y/N)?". If so, type `Y` and press Enter.
4.  The server will stop, and you can close the terminal window. The web page in your browser will no longer work until you run `python app.py` again.

---
