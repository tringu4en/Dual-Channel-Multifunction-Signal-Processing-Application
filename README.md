# Dual Channel Multifunction Signal Processing Application

## Project Overview
This project is a robust desktop application architected using **PySide6** for the real-time visualization and manipulation of biomedical and audio signals. It leverages **NumPy** and **SciPy** to perform high-speed Fast Fourier Transforms (FFT) and digital filtering with a sampling rate tolerance of 1-0.1 Hz. 

The system features a custom-built data-logging engine capable of continuous, long-term recording via USB serial communication without system latency.

## Key Features

### 1. Dual Mode Operation
* **Real-time Mode:** Connects to embedded devices (e.g., ESP32, STM32) via Serial Port to visualize live `RMS_Ch1, RMS_Ch2` data streams.
* **File Mode:** Supports loading and processing Mono or Stereo `.wav` files for offline analysis.

### 2. Advanced Signal Processing
* **Digital Filtering:** Implements 5th-order Butterworth Low-pass and High-pass filters, adjustable in real-time.
* **FFT Analysis:** Computes Magnitude Spectrum using a Hanning window to reduce spectral leakage, applying a $4/N$ correction factor for accurate energy representation.

### 3. User Interface & Visualization
* **Interactive Plotting:** High-performance rendering using `pyqtgraph`. Includes "Clickable Plots" that open a detailed **ZoomDialog** for granular signal inspection.
* **Dynamic Controls:** Real-time toggles for filtering, manual sampling rate ($F_s$) overrides, and serial connection management.

### 4. Data Logging
* **Buffered Recording:** Writes data to CSV files in chunks to ensure thread safety and prevent I/O blocking during high-speed transmission.

## Project Structure

* `main.py`: The application entry point. Launches the Rules Dialog followed by the Main Window.
* `class_side.py`: Contains the core `MainWindow` logic, UI layout, serial communication handling, and event loops.
* `signal_processor.py`: The math engine handling filter design (Butterworth) and FFT computation (normalization and windowing).
* `startup.py`: A modal dialog that enforces user acknowledgement of data formats and rules before operation.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YourUsername/Dual-Channel-Signal-Processor.git](https://github.com/YourUsername/Dual-Channel-Signal-Processor.git)
    ```
2.  **Install dependencies:**
    ```bash
    pip install numpy scipy PySide6 pyqtgraph
    ```

## Usage Guide

1.  **Launch the Application:**
    ```bash
    python main.py
    ```
2.  **Startup Rules:**
    * Upon launch, acknowledge the "Rules and Announcements" dialog.
3.  **Real-time Connection:**
    * Ensure your microcontroller sends data in the format: `RMS_Ch1,RMS_Ch2` (comma-separated values).
    * Select the correct **COM Port** and **Baud Rate** (Default: 921600).
    * Click **Initiate** to detect the sampling rate, then **Start Recording**.
4.  **File Analysis:**
    * Select "From Audio File" mode.
    * Load a `.wav` file to instantly visualize Raw Signal, Filtered Signal, and FFT.

## Requirements
* Python 3.x
* PySide6
* NumPy
* SciPy
* PyQtGraph
