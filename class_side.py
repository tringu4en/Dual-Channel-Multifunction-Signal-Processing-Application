# --- Imports ---
# Standard library imports
import sys
import time # Import time module
import csv
import tempfile # For creating temporary files
import os       # For file operations (rename, remove)


from signal_processor import SignalProcessor

# Third-party imports
import numpy as np
from scipy import signal
from scipy.io import wavfile # For reading WAV files
# import serial # Not used, commented out in original
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                               QPushButton, QComboBox, QMainWindow,
                               QGridLayout, QLineEdit,
                               QMessageBox, QFileDialog, QGroupBox, QRadioButton, QDialog)
from PySide6.QtSerialPort import QSerialPort, QSerialPortInfo
from PySide6.QtCore import QIODevice, Qt, Signal
from pyqtgraph import PlotWidget, mkPen
from scipy.fft import fft, fftfreq

# --- Constants ---
INITIATION_SAMPLES = 550  # Number of data points (lines) to collect for initiation
REQUIRED_INTERVALS = INITIATION_SAMPLES // 2  # Min number of time intervals needed
INITIATION_TIMEOUT_S = 10  # Max seconds to wait for initiation samples
MAX_BUFFER_SIZE = 2048  # Max samples (RMS value pairs) to keep in buffer for real-time display
UPDATE_THRESHOLD = 100  # Update plots after this many new lines (RMS value pairs)
DEFAULT_HIGHCUT = 10000.0  # Default high cutoff frequency (less relevant for RMS, but kept for consistency)
FILTER_ORDER = 5  # Butterworth filter order
MIN_VALID_INTERVAL = 1e-7  # Smallest realistic time interval (adjust if needed)
CSV_BUFFER_SIZE = 100  # Write to file every 100 points
MAX_MANUAL_FREQ_INPUT = 22000.0  # Max frequency for manual input via QLineEdit
MIN_MANUAL_FS_INPUT = 1.0  # Minimum reasonable Fs (rate of RMS value pairs)

class ClickablePlotWidget(PlotWidget):
    """A PlotWidget that emits a signal when double-clicked."""
    sigDoubleClicked = Signal(object)  # Emits reference to self

    def mouseDoubleClickEvent(self, event):
        # Emit signal before doing default behavior
        self.sigDoubleClicked.emit(self) 
        super().mouseDoubleClickEvent(event)

class ZoomDialog(QDialog):
    """The popup window to display the zoomed plot."""
    def __init__(self, title, x_label, y_label, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Zoomed View: {title}")
        self.resize(900, 600) # Default popup size
        
        layout = QVBoxLayout(self)
        self.plot_widget = PlotWidget()
        self.plot_widget.setTitle(title)
        self.plot_widget.setLabel("bottom", x_label)
        self.plot_widget.setLabel("left", y_label)
        self.plot_widget.showGrid(x=True, y=True)
        
        layout.addWidget(self.plot_widget)

    def update_data(self, x, y, pen_color):
        """Updates the plot data in real-time."""
        self.plot_widget.plot(x, y, pen=mkPen(pen_color), clear=True)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_variables()
        self.init_ui()

    def init_variables(self):
        # Data storage for Channel 1
        self.data_array_raw_ch1 = np.array([])
        self.data_array_fined_ch1 = np.array([]) # May not be used if raw is processed directly
        self.fft_data_ch1 = np.array([])
        self.freq_ch1 = np.array([])
        self.filtered_data_ch1 = np.array([])

        # Data storage for Channel 2
        self.data_array_raw_ch2 = np.array([])
        self.data_array_fined_ch2 = np.array([]) # May not be used
        self.fft_data_ch2 = np.array([])
        self.freq_ch2 = np.array([])
        self.filtered_data_ch2 = np.array([])

        # Initiation phase temporary storage (now stores tuples or is handled differently)
        self.time_interval_array = np.array([])
        # self.data_array_raw_test_ch1 = np.array([]) # Not strictly needed if we don't plot during init
        # self.data_array_raw_test_ch2 = np.array([])

        self.number_of_data_lines = 0 # Counts pairs of data points (lines)
        self.last_read_time = time.perf_counter()
        self.initiating_data_count = 0
        self.initiation_start_time = 0

        self.lowcut = 0.0
        self.highcut = DEFAULT_HIGHCUT
        self.filter_order = FILTER_ORDER

        self.is_initiating = False
        self.is_ready_to_record = False
        self.is_recording = False
        self.continuing_plot_after_stop = False
        self.prompt_save_on_disconnect_after_stop = False
        self.manual_fs_override = False

        self.fft_mode = "realtime" # "realtime" or "file"
        self.audio_file_path = None
        self.audio_file_data_ch1 = np.array([]) # For WAV Ch1
        self.audio_file_data_ch2 = np.array([]) # For WAV Ch2
        self.audio_file_sampling_rate = None

        self.main_time_interval = 0.0
        self.sampling_rate = None # Rate of data pairs (realtime) or audio Fs (file)

        self.sliders_enabled = False
        self.serial = QSerialPort()

        self.csv_temp_file_path = None
        self.csv_file = None
        self.csv_writer = None
        self.csv_buffer = []
        self.recording_start_time = 0

        self.signal_processor = SignalProcessor()

        self.last_plot_update_time = time.perf_counter()

        self.current_zoom_window = None
        self.zoomed_plot_source = None

        print("Variables Initialized for Dual Channel")

    def init_ui(self):
        self.setWindowTitle("Dual Channel Sound Signal Processor")
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_h_layout = QHBoxLayout(central_widget)

        left_pane_widget = QWidget()
        left_pane_layout = QVBoxLayout(left_pane_widget)
        right_pane_widget = QWidget()
        right_pane_layout = QVBoxLayout(right_pane_widget)

        # == Plot Widgets for Left Pane (Dual Channel Layout) ==
        # --- Raw Signal Plots ---
        raw_plots_group = QGroupBox("Raw Signal Display (RMS for Real-time, Audio for File)")
        raw_plots_layout = QHBoxLayout()
        self.raw_signal_plot_ch1 = ClickablePlotWidget()
        self.raw_signal_plot_ch1.setTitle("Channel 1")
        self.raw_signal_plot_ch1.setLabel("left", "Value")
        self.raw_signal_plot_ch1.setLabel("bottom", "Sample Index")
        self.raw_signal_plot_ch1.showGrid(x=True, y=True)
        self.raw_signal_plot_ch1.sigDoubleClicked.connect(self.open_zoom_window) # CONNECT SIGNAL
        raw_plots_layout.addWidget(self.raw_signal_plot_ch1)

        self.raw_signal_plot_ch2 = ClickablePlotWidget()
        self.raw_signal_plot_ch2.setTitle("Channel 2")
        self.raw_signal_plot_ch2.setLabel("left", "Value")
        self.raw_signal_plot_ch2.setLabel("bottom", "Sample Index")
        self.raw_signal_plot_ch2.showGrid(x=True, y=True)
        self.raw_signal_plot_ch2.sigDoubleClicked.connect(self.open_zoom_window) # CONNECT SIGNAL
        raw_plots_layout.addWidget(self.raw_signal_plot_ch2)
        raw_plots_group.setLayout(raw_plots_layout)
        left_pane_layout.addWidget(raw_plots_group)

        # --- Filtered Signal Plots ---
        filtered_plots_group = QGroupBox("Filtered Signal Display")
        filtered_plots_layout = QHBoxLayout()
        self.filtered_signal_plot_ch1 = ClickablePlotWidget()
        self.filtered_signal_plot_ch1.setTitle("Channel 1")
        self.filtered_signal_plot_ch1.setLabel("left", "Value (Filtered)")
        self.filtered_signal_plot_ch1.setLabel("bottom", "Sample Index")
        self.filtered_signal_plot_ch1.showGrid(x=True, y=True)
        filtered_plots_layout.addWidget(self.filtered_signal_plot_ch1)
        self.filtered_signal_plot_ch1.sigDoubleClicked.connect(self.open_zoom_window) # CONNECT SIGNAL
        raw_plots_layout.addWidget(self.filtered_signal_plot_ch1)

        self.filtered_signal_plot_ch2 = ClickablePlotWidget()
        self.filtered_signal_plot_ch2.setTitle("Channel 2")
        self.filtered_signal_plot_ch2.setLabel("left", "Value (Filtered)")
        self.filtered_signal_plot_ch2.setLabel("bottom", "Sample Index")
        self.filtered_signal_plot_ch2.showGrid(x=True, y=True)
        filtered_plots_layout.addWidget(self.filtered_signal_plot_ch2)
        self.filtered_signal_plot_ch2.sigDoubleClicked.connect(self.open_zoom_window) # CONNECT SIGNAL
        raw_plots_layout.addWidget(self.filtered_signal_plot_ch2)
        filtered_plots_group.setLayout(filtered_plots_layout)
        left_pane_layout.addWidget(filtered_plots_group)

        # --- FFT Plots ---
        fft_plots_group = QGroupBox("FFT (Magnitude Spectrum)")
        fft_plots_layout = QHBoxLayout()
        self.fft_plot_ch1 = ClickablePlotWidget()
        self.fft_plot_ch1.setTitle("Channel 1")
        self.fft_plot_ch1.setLabel("left", "Magnitude")
        self.fft_plot_ch1.setLabel("bottom", "Frequency (Hz)")
        self.fft_plot_ch1.showGrid(x=True, y=True)
        self.fft_plot_ch1.setLogMode(x=False, y=False)
        fft_plots_layout.addWidget(self.fft_plot_ch1)
        self.fft_plot_ch1.sigDoubleClicked.connect(self.open_zoom_window) # CONNECT SIGNAL
        raw_plots_layout.addWidget(self.fft_plot_ch1)

        self.fft_plot_ch2 = ClickablePlotWidget()
        self.fft_plot_ch2.setTitle("Channel 2")
        self.fft_plot_ch2.setLabel("left", "Magnitude")
        self.fft_plot_ch2.setLabel("bottom", "Frequency (Hz)")
        self.fft_plot_ch2.showGrid(x=True, y=True)
        self.fft_plot_ch2.setLogMode(x=False, y=False)
        fft_plots_layout.addWidget(self.fft_plot_ch2)
        self.fft_plot_ch2.sigDoubleClicked.connect(self.open_zoom_window) # CONNECT SIGNAL
        raw_plots_layout.addWidget(self.fft_plot_ch2)
        fft_plots_group.setLayout(fft_plots_layout)
        left_pane_layout.addWidget(fft_plots_group)

        # == Elements for Right Pane (Controls) ==
        # (Control elements remain largely the same as in the original file)
        # ... (Copy and paste the right_pane_layout setup from your original file here) ...
        # Make sure to connect signals to the new or modified handlers

        fft_mode_group = QGroupBox("FFT Source Mode")
        fft_mode_layout = QVBoxLayout()
        self.realtime_fft_radio = QRadioButton("Real-time (Serial Port - Dual RMS)")
        self.realtime_fft_radio.setChecked(True)
        self.realtime_fft_radio.toggled.connect(self.handle_fft_mode_change)
        self.file_fft_radio = QRadioButton("From Audio File (.wav - Mono/Stereo)")
        self.file_fft_radio.toggled.connect(self.handle_fft_mode_change)

        file_browse_layout = QHBoxLayout()
        self.browse_audio_button = QPushButton("Browse...")
        self.browse_audio_button.clicked.connect(self.handle_browse_audio_file)
        self.browse_audio_button.setEnabled(False)
        self.selected_file_label = QLabel("No file selected.")
        self.selected_file_label.setWordWrap(True)
        file_browse_layout.addWidget(self.browse_audio_button)
        file_browse_layout.addWidget(self.selected_file_label, 1)

        fft_mode_layout.addWidget(self.realtime_fft_radio)
        fft_mode_layout.addWidget(self.file_fft_radio)
        fft_mode_layout.addLayout(file_browse_layout)
        fft_mode_group.setLayout(fft_mode_layout)
        right_pane_layout.addWidget(fft_mode_group)

        connection_group = QGroupBox("Serial Connection")
        connection_layout = QVBoxLayout(connection_group)

        com_port_widget = QWidget()
        com_port_layout = QVBoxLayout(com_port_widget)
        self.com_label = QLabel("COM Port:")
        self.port_select = QComboBox(self)
        self.baud_label = QLabel("Baud Rate:")
        self.baud_combobox = QComboBox()
        self.baud_combobox.addItems(["115200", "500000","921600","1000000","2000000"])
        default_item = "921600" # Match ESP32 code
        index = self.baud_combobox.findText(default_item)
        if index != -1:
            self.baud_combobox.setCurrentIndex(index)

        fs_layout = QHBoxLayout()
        self.fs_label = QLabel("Manual Fs (Hz):")
        self.fs_input = QLineEdit()
        self.fs_input.setPlaceholderText("e.g., 100 (RMS pairs/sec)")
        self.fs_input.setMaximumWidth(100)
        self.apply_fs_button = QPushButton("Apply Fs")
        self.apply_fs_button.clicked.connect(self.handle_apply_fs_button)
        fs_layout.addWidget(self.fs_label)
        fs_layout.addWidget(self.fs_input)
        fs_layout.addWidget(self.apply_fs_button)
        fs_layout.addStretch(1)

        self.status_label = QLabel("STATUS: Disconnected", self)
        self.status_label.setStyleSheet("color: #FF0000; font-weight: bold;")
        self.status_label.setWordWrap(True)

        com_port_layout.addWidget(self.com_label)
        com_port_layout.addWidget(self.port_select)
        com_port_layout.addWidget(self.baud_label)
        com_port_layout.addWidget(self.baud_combobox)
        com_port_layout.addLayout(fs_layout)
        com_port_layout.addWidget(self.status_label)

        button_widget = QWidget()
        button_layout = QGridLayout(button_widget)
        self.initiate_start_button = QPushButton("Initiate", self)
        self.initiate_start_button.setToolTip("Connect to port and determine data rate, then click again to start recording.")
        self.initiate_start_button.clicked.connect(self.handle_initiate_start_button)
        self.refresh_button = QPushButton("Refresh Ports", self)
        self.refresh_button.clicked.connect(self.update_com_ports)

        self.disconnect_button = QPushButton("Disconnect", self)
        self.disconnect_button.setEnabled(False)
        self.disconnect_button.clicked.connect(self.handle_save_data_button) 
        self.disconnect_button.clicked.connect(self.disconnect_serial)      

        self.reset_button = QPushButton("Reset All", self)
        self.reset_button.setToolTip("Reset connection, data, and filters")
        self.reset_button.clicked.connect(self.reset_alldata)

        self.clear_plots_button = QPushButton("Clear Plots", self)
        self.clear_plots_button.setToolTip("Clear data arrays and plots")
        self.clear_plots_button.clicked.connect(self.handle_save_data_button)
        self.clear_plots_button.clicked.connect(self.clear_plot_data)

        self.stop_recording_button = QPushButton("Stop Recording", self)
        self.stop_recording_button.setToolTip("Stop the current recording and prompt to save.")
        self.stop_recording_button.clicked.connect(self.handle_stop_recording_button)
        self.stop_recording_button.setEnabled(False)

        self.save_data_button = QPushButton("Save Data", self)
        self.save_data_button.setToolTip("Save the currently recorded data to a CSV file.")
        self.save_data_button.clicked.connect(self.handle_save_data_button)
        self.save_data_button.setEnabled(False)

        button_layout.addWidget(self.refresh_button, 0, 0)
        button_layout.addWidget(self.initiate_start_button, 0, 1)
        button_layout.addWidget(self.reset_button, 1, 0)
        button_layout.addWidget(self.clear_plots_button, 1, 1)
        button_layout.addWidget(self.disconnect_button, 2, 0)
        button_layout.addWidget(self.stop_recording_button, 2, 1)
        button_layout.addWidget(self.save_data_button, 3, 0, 1, 2)

        connection_layout.addWidget(com_port_widget)
        connection_layout.addWidget(button_widget)
        right_pane_layout.addWidget(connection_group)

        filter_input_group = QGroupBox("Filter Settings")
        filter_input_widget = QWidget()
        filter_input_layout = QHBoxLayout(filter_input_widget)
        self.Min_value_label = QLabel(f"Low Cut: {self.lowcut:.0f} Hz")
        self.number_input1 = QLineEdit(str(int(self.lowcut)))
        self.number_input1.setPlaceholderText("Min (Hz)")
        self.number_input1.setMaximumWidth(80)
        self.number_input1.textChanged.connect(self.on_text1_changed)
        self.Max_value_label = QLabel(f"High Cut: {self.highcut:.0f} Hz")
        self.number_input2 = QLineEdit(str(int(self.highcut)))
        self.number_input2.setPlaceholderText("Max (Hz)")
        self.number_input2.setMaximumWidth(80)
        self.number_input2.textChanged.connect(self.on_text2_changed)
        filter_input_layout.addWidget(self.Min_value_label)
        filter_input_layout.addWidget(self.number_input1)
        filter_input_layout.addSpacing(20)
        filter_input_layout.addWidget(self.Max_value_label)
        filter_input_layout.addWidget(self.number_input2)
        filter_input_layout.addStretch(1)
        filter_input_group.setLayout(filter_input_layout)
        right_pane_layout.addWidget(filter_input_group)

        filter_button_widget = QWidget()
        filter_button_layout = QVBoxLayout(filter_button_widget)
        self.toggle_button = QPushButton("Enable Filtering")
        self.toggle_button.setCheckable(True)
        self.toggle_button.setChecked(self.sliders_enabled)
        self.toggle_button.clicked.connect(self.toggle_sliders)
        self.reset_filters_button = QPushButton("Reset Filters")
        self.reset_filters_button.clicked.connect(self.reset_sliders)
        filter_button_layout.addWidget(self.toggle_button)
        filter_button_layout.addWidget(self.reset_filters_button)
        right_pane_layout.addWidget(filter_button_widget)

        right_pane_layout.addStretch(1)


        main_h_layout.addWidget(left_pane_widget, 3) # Left pane takes 3/4 width
        main_h_layout.addWidget(right_pane_widget, 1) # Right pane takes 1/4 width

        self.toggle_sliders()
        self.update_com_ports()
        self.update_fs_input_state()
        self.handle_fft_mode_change() # Initial setup based on mode
        self.update_save_button_state()
        print("UI Initialized with Dual Channel Plot Layout.")

    def handle_fft_mode_change(self):
        if self.realtime_fft_radio.isChecked():
            self.fft_mode = "realtime"
            print("FFT Mode: Real-time (Dual RMS)")
            if self.audio_file_path: # Clear any loaded file data
                self.audio_file_path = None
                self.audio_file_data_ch1 = np.array([])
                self.audio_file_data_ch2 = np.array([])
                self.audio_file_sampling_rate = None
                self.selected_file_label.setText("No file selected.")
                self.clear_plot_data(prompt_save=True) # Clear plots and data arrays

            self.port_select.setEnabled(True)
            self.baud_combobox.setEnabled(True)
            self.refresh_button.setEnabled(True)
            self.fs_input.setReadOnly(False)
            self.apply_fs_button.setEnabled(True)
            self.browse_audio_button.setEnabled(False)

            self.sampling_rate = None # Will be determined or manually set
            self.manual_fs_override = False
            self.fs_input.setText("")
            self.update_fs_input_state()
            self.reset_to_disconnected_state()

        elif self.file_fft_radio.isChecked():
            self.fft_mode = "file"
            print("FFT Mode: From Audio File (Mono/Stereo)")
            if self.is_recording:
                save_result = self._trigger_save_if_needed(context="mode_switch_to_file")
                if save_result == "cancel":
                    self.realtime_fft_radio.setChecked(True); return
                if self.is_recording: self._internal_stop_recording_logic(ask_to_save=False)
            if self.serial.isOpen(): self.disconnect_serial(prompt_save=False)

            self.port_select.setEnabled(False)
            self.baud_combobox.setEnabled(False)
            self.refresh_button.setEnabled(False)
            self.initiate_start_button.setEnabled(False)
            self.disconnect_button.setEnabled(False)
            self.stop_recording_button.setEnabled(False)
            self.fs_input.setReadOnly(True) # Fs comes from file
            self.apply_fs_button.setEnabled(False)
            self.browse_audio_button.setEnabled(True)
            self.status_label.setText("STATUS: Select an audio file for FFT.")
            self.status_label.setStyleSheet("color: #0000FF;")
        
        self.update_fs_input_state()
        self.update_save_button_state()
        self.update_plots() # Update plots to reflect mode change (e.g., clear or show file data)


    def handle_browse_audio_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Audio File", "", "WAV Files (*.wav);;All Files (*)")
        if file_path:
            print(f"Audio file selected: {file_path}")
            self.load_audio_file(file_path)

    def load_audio_file(self, file_path):
        try:
            fs, data = wavfile.read(file_path)
            print(f"File loaded: {file_path}, Fs: {fs}, Samples: {data.shape}")

            self.audio_file_path = file_path
            self.audio_file_sampling_rate = float(fs)
            self.sampling_rate = self.audio_file_sampling_rate # Set sampling rate for file mode
            self.manual_fs_override = True # Fs is fixed by file

            # Clear previous raw data arrays for channels
            self.data_array_raw_ch1 = np.array([])
            self.data_array_raw_ch2 = np.array([])

            if data.ndim == 1: # Mono file
                print("Mono file detected.")
                self.audio_file_data_ch1 = data.astype(np.float32)
                self.audio_file_data_ch2 = np.array([]) # No data for channel 2
            elif data.ndim > 1 and data.shape[1] >= 2: # Stereo or more channels
                print("Stereo/multi-channel file detected, using first two channels.")
                self.audio_file_data_ch1 = data[:, 0].astype(np.float32)
                self.audio_file_data_ch2 = data[:, 1].astype(np.float32)
            else: # Should not happen for valid WAV
                print("Warning: Could not determine channel data from WAV.")
                self.audio_file_data_ch1 = np.array([])
                self.audio_file_data_ch2 = np.array([])
                raise ValueError("Unsupported WAV file format or channel structure.")

            # Normalize audio file data (example, per channel)
            if self.audio_file_data_ch1.size > 0:
                max_val_ch1 = np.max(np.abs(self.audio_file_data_ch1))
                if max_val_ch1 > 0: self.audio_file_data_ch1 /= max_val_ch1
            if self.audio_file_data_ch2.size > 0:
                max_val_ch2 = np.max(np.abs(self.audio_file_data_ch2))
                if max_val_ch2 > 0: self.audio_file_data_ch2 /= max_val_ch2
            
            self.fs_input.setText(f"{self.sampling_rate:.1f}")
            self.fs_input.setReadOnly(True)
            self.apply_fs_button.setEnabled(False)

            self.selected_file_label.setText(os.path.basename(file_path))
            self.status_label.setText(f"STATUS: Audio file loaded. Fs: {self.sampling_rate:.1f} Hz")
            self.status_label.setStyleSheet("color: #008000;")

            self.update_plots() # This will now use audio_file_data_ch1/ch2
            self.reset_sliders() # Reset filters for new data characteristics

        except FileNotFoundError:
            QMessageBox.critical(self, "Error", f"File not found: {file_path}")
            self.selected_file_label.setText("Error: File not found.")
        except Exception as e:
            QMessageBox.critical(self, "Error Loading File", f"Could not load or process audio file:\n{e}")
            print(f"Error loading audio file {file_path}: {e}")
            self.selected_file_label.setText("Error loading file.")
            self.audio_file_path = None
            self.audio_file_data_ch1 = np.array([])
            self.audio_file_data_ch2 = np.array([])
            self.audio_file_sampling_rate = None
        self.update_save_button_state()


    def update_com_ports(self):
        # (Largely same as original, ensure UI elements are enabled/disabled correctly based on fft_mode)
        current_port = self.port_select.currentText()
        self.port_select.clear()
        available_ports = QSerialPortInfo.availablePorts()
        ports_found = False
        for port_info in available_ports: # Renamed port to port_info to avoid conflict
            self.port_select.addItem(port_info.portName())
            if port_info.portName() == current_port:
                 self.port_select.setCurrentText(current_port)
            ports_found = True

        self.port_select.setEnabled(ports_found)
        if not ports_found:
            self.port_select.addItem("No ports found")
            self.initiate_start_button.setEnabled(False)
        else:
            if self.fft_mode == "realtime":
                 self.initiate_start_button.setEnabled(not self.is_initiating and not self.is_ready_to_record and not self.is_recording)
            else: # File mode
                 self.initiate_start_button.setEnabled(False)
        
        self.stop_recording_button.setEnabled(self.is_recording)
        self.update_fs_input_state()
        self.update_save_button_state()
        print("COM Ports Updated")


    def handle_initiate_start_button(self):
        if self.fft_mode == "file":
            QMessageBox.information(self, "Mode Info", "Real-time recording is disabled in 'From Audio File' mode.")
            return

        if not self.is_initiating and not self.is_ready_to_record and not self.is_recording: # Initiate
            print("Initiate button clicked.")
            if self.serial.isOpen(): self.serial.close()
            selected_port = self.port_select.currentText()
            if not selected_port or selected_port == "No ports found":
                self.status_label.setText("STATUS: No port selected"); self.status_label.setStyleSheet("color: #FFA500;")
                return
            baud_rate = int(self.baud_combobox.currentText())
            self.serial.setPortName(selected_port); self.serial.setBaudRate(baud_rate)

            print(f"Attempting to connect to {selected_port} at {baud_rate} baud for initiation...")
            if self.serial.open(QIODevice.OpenModeFlag.ReadWrite):
                self.serial.clear(QSerialPort.Direction.Input)
                self.status_label.setText(f"STATUS: Connected. Initiating (0/{INITIATION_SAMPLES} lines)..."); self.status_label.setStyleSheet("color: #FFFF00;")
                self.initiate_start_button.setEnabled(False); self.disconnect_button.setEnabled(True); self.stop_recording_button.setEnabled(False)
                self.port_select.setEnabled(False); self.baud_combobox.setEnabled(False); self.refresh_button.setEnabled(False)
                self.manual_fs_override = False; self.fs_input.setText("")

                self.is_initiating = True; self.is_ready_to_record = False; self.is_recording = False
                self.continuing_plot_after_stop = False; self.prompt_save_on_disconnect_after_stop = False
                self.initiating_data_count = 0; self.time_interval_array = np.array([])
                self.sampling_rate = None; self.last_read_time = time.perf_counter(); self.initiation_start_time = time.perf_counter()
                self.update_fs_input_state()
                try: self.serial.readyRead.connect(self.initiating_read_dual_channel)
                except Exception as e: print(f"Error connecting signal: {e}"); self.disconnect_serial(prompt_save=False); return
            else:
                self.status_label.setText(f"STATUS: Failed to connect: {self.serial.errorString()}"); self.status_label.setStyleSheet("color: #FF0000;")
                self.reset_to_disconnected_state()

        elif self.is_ready_to_record and not self.is_recording: # Start Recording
            print("Start Recording button clicked.")
            if not self.serial.isOpen():
                 self.status_label.setText("STATUS: Error - Port Closed"); self.status_label.setStyleSheet("color: #FF0000;")
                 self.reset_to_disconnected_state(); return
            if self.sampling_rate is None or self.sampling_rate < MIN_MANUAL_FS_INPUT:
                QMessageBox.warning(self, "Fs Not Set", "Sampling frequency (Fs) is not set or is invalid. Please initiate or apply a manual Fs.")
                self.is_ready_to_record = True; self.update_fs_input_state(); return

            self.is_ready_to_record = False; self.is_recording = True
            if self.csv_temp_file_path and os.path.exists(self.csv_temp_file_path):
                try: os.remove(self.csv_temp_file_path)
                except Exception as e: print(f"Error removing previous temp CSV: {e}")
            self.csv_temp_file_path = None; self.prompt_save_on_disconnect_after_stop = False

            if not self.continuing_plot_after_stop: self.clear_plot_data(prompt_save=False) # Clear data for new recording
            else: print("Continuing plot data from previous stop.")
            self.continuing_plot_after_stop = False

            try: # Setup CSV for dual channel
                temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, newline='', suffix='.csv', prefix='recording_')
                self.csv_temp_file_path = temp_file.name; self.csv_file = temp_file; self.csv_writer = csv.writer(self.csv_file)
                self.csv_writer.writerow(['Timestamps', 'RMS_Ch1', 'RMS_Ch2']); self.csv_buffer = []; self.recording_start_time = time.perf_counter()
                print(f"Temporary CSV for recording: {self.csv_temp_file_path}")
            except Exception as e:
                QMessageBox.warning(self, "CSV Error", f"Could not create temp CSV:\n{e}") # ... (handle error) ...
                self.csv_temp_file_path = None; self.csv_file = None; self.csv_writer = None

            self.initiate_start_button.setEnabled(False); self.disconnect_button.setEnabled(True); self.stop_recording_button.setEnabled(True)
            self.update_fs_input_state()
            status_fs_text = f"(Manual Fs: {self.sampling_rate:.1f} Hz)" if self.manual_fs_override else f"(Fs: {self.sampling_rate:.1f} Hz)"
            if self.csv_writer: self.status_label.setText(f"STATUS: Recording {status_fs_text}"); self.status_label.setStyleSheet("color: #00DD00;")
            else: self.status_label.setText(f"STATUS: Recording (CSV Error!) {status_fs_text}"); self.status_label.setStyleSheet("color: #FFA500;")
            
            self.last_read_time = time.perf_counter(); self.number_of_data_lines = 0
            try: self.serial.readyRead.connect(self.read_data_dual_channel)
            except Exception as e:
                 print(f"Error connecting read_data signal: {e}"); self._internal_stop_recording_logic(ask_to_save=True, triggered_by_error=True)
                 self.status_label.setText("STATUS: Error starting recording."); self.status_label.setStyleSheet("color: #FF0000;"); return
        else:
            print(f"Warning: Initiate/Start button in unexpected state: init={self.is_initiating}, ready={self.is_ready_to_record}, rec={self.is_recording}")
        self.update_save_button_state()

    def initiating_read_dual_channel(self):
        if not self.is_initiating or not self.serial.isOpen() or not self.serial.canReadLine():
            if self.is_initiating and not self.serial.isOpen(): self.disconnect_serial(prompt_save=False); return
            return
        if time.perf_counter() - self.initiation_start_time > INITIATION_TIMEOUT_S:
            self.status_label.setText("STATUS: Initiation Timeout"); self.status_label.setStyleSheet("color: #FF0000;")
            self.disconnect_serial(prompt_save=False); return

        while self.serial.canReadLine() and self.initiating_data_count < INITIATION_SAMPLES:
            if not self.is_initiating: return
            try:
                line_bytes = self.serial.readLine()
                if not line_bytes: continue # Should not happen with canReadLine but good check
                line = line_bytes.data().decode('utf-8', errors='ignore').strip()
                
                current_time = time.perf_counter(); time_diff = current_time - self.last_read_time; self.last_read_time = current_time
                if line:
                    parts = line.split(',')
                    if len(parts) == 2: # Expecting two values
                        # We don't necessarily need to store/plot these initial values, just time them
                        # data_point_ch1 = float(parts[0])
                        # data_point_ch2 = float(parts[1])
                        if self.initiating_data_count > 0 and time_diff > MIN_VALID_INTERVAL: # After the first data point
                            self.time_interval_array = np.append(self.time_interval_array, time_diff)
                        self.initiating_data_count += 1
                        self.status_label.setText(f"STATUS: Initiating ({self.initiating_data_count}/{INITIATION_SAMPLES} lines)...")
                    else:
                        print(f"Warning: Malformed line during initiation: {line}")
                        continue # Skip malformed line
            except ValueError: 
                print(f"Warning: ValueError for line during initiation: {line}")
                continue
            except Exception as e:
                self.status_label.setText(f"STATUS: Read Error during Init"); self.status_label.setStyleSheet("color: #FF0000;");
                self.disconnect_serial(prompt_save=False); return

            if self.initiating_data_count >= INITIATION_SAMPLES:
                try: self.serial.readyRead.disconnect(self.initiating_read_dual_channel)
                except (RuntimeError, TypeError): pass

                if len(self.time_interval_array) >= REQUIRED_INTERVALS:
                    self.main_time_interval = np.median(self.time_interval_array)
                    if self.main_time_interval >= MIN_VALID_INTERVAL:
                        self.sampling_rate = 1.0 / self.main_time_interval
                        self.is_initiating = False; self.is_ready_to_record = True; self.continuing_plot_after_stop = False; self.prompt_save_on_disconnect_after_stop = False
                        self.manual_fs_override = False; self.fs_input.setText(f"{self.sampling_rate:.1f}")
                        self.status_label.setText(f"STATUS: Ready (Fs: {self.sampling_rate:.1f} lines/s)"); self.status_label.setStyleSheet("color: #00FF00;")
                        self.initiate_start_button.setText("Start Recording"); self.initiate_start_button.setEnabled(True)
                        self.disconnect_button.setEnabled(True); self.stop_recording_button.setEnabled(False)
                        self.port_select.setEnabled(False); self.baud_combobox.setEnabled(False); self.refresh_button.setEnabled(False)
                        self.update_fs_input_state(); self.time_interval_array = np.array([])
                        self.clear_plot_data(prompt_save=False); # Clear plots before starting
                        self.reset_sliders()
                        return
                    else: self.status_label.setText("STATUS: Initiation Error (Interval Calc)")
                else: self.status_label.setText("STATUS: Initiation Error (Not enough intervals)")
                self.status_label.setStyleSheet("color: #FF0000;"); self.disconnect_serial(prompt_save=False); return
        self.update_save_button_state()


    def read_data_dual_channel(self):
        if self.fft_mode == "file": return
        if not self.is_recording or not self.serial.isOpen() or not self.serial.canReadLine():
             if self.is_recording and not self.serial.isOpen():
                  self._internal_stop_recording_logic(ask_to_save=True, triggered_by_error=True)
                  self.status_label.setText("STATUS: Port Error! Rec Stopped."); self.status_label.setStyleSheet("color: #FF0000;")
                  return
             return

        processed_lines = 0
        # Read multiple lines if available, up to a limit to keep UI responsive
        while self.serial.canReadLine() and processed_lines < 200: # Process up to 200 lines per call
            if not self.is_recording: return
            try:
                line_bytes = self.serial.readLine()
                if not line_bytes: continue
                line = line_bytes.data().decode('utf-8', errors='ignore').strip()

                if line:
                    parts = line.split(',')
                    if len(parts) == 2:
                        rms_ch1 = float(parts[0])
                        rms_ch2 = float(parts[1])
                        
                        self.data_array_raw_ch1 = np.append(self.data_array_raw_ch1, rms_ch1)
                        self.data_array_raw_ch2 = np.append(self.data_array_raw_ch2, rms_ch2)
                        self.number_of_data_lines += 1
                        processed_lines += 1

                        if self.csv_writer:
                            try:
                                self.csv_buffer.append([time.perf_counter() - self.recording_start_time, rms_ch1, rms_ch2])
                                if len(self.csv_buffer) >= CSV_BUFFER_SIZE: self.flush_csv_buffer()
                            except Exception as csv_e: # ... (handle CSV error) ...
                                print(f"CSV Write Error: {csv_e}") # Basic error logging
                                self._internal_stop_recording_logic(ask_to_save=True, triggered_by_error=True)
                                self.status_label.setText("STATUS: CSV Write Error! Rec Stopped."); self.status_label.setStyleSheet("color: #FFA500;")
                                return
                    else:
                        print(f"Warning: Malformed line during recording: {line}")
                        continue # Skip malformed line
            except ValueError: 
                print(f"Warning: ValueError for line during recording: {line}")
                continue # Skip malformed line
            except Exception as e:
                self._internal_stop_recording_logic(ask_to_save=True, triggered_by_error=True)
                self.status_label.setText(f"STATUS: Read Err! Rec Stopped."); self.status_label.setStyleSheet("color: #FF0000;");
                return

        # Trim buffers
        if len(self.data_array_raw_ch1) > MAX_BUFFER_SIZE:
            self.data_array_raw_ch1 = self.data_array_raw_ch1[-MAX_BUFFER_SIZE:]
        if len(self.data_array_raw_ch2) > MAX_BUFFER_SIZE:
            self.data_array_raw_ch2 = self.data_array_raw_ch2[-MAX_BUFFER_SIZE:]
        
        # Update plots if enough new data lines have arrived
        if self.number_of_data_lines >= UPDATE_THRESHOLD:
            # --- NEW: Fs Auto-Calibration Logic ---
            current_time = time.perf_counter()
            time_elapsed = current_time - self.last_plot_update_time
            
            # Avoid division by zero
            if time_elapsed > 0:
                # Calculate instant Fs: (Samples received / Time taken)
                instant_fs = self.number_of_data_lines / time_elapsed
                
                # Apply "Smoothing" so the number doesn't jump wildly
                # 90% old value, 10% new value
                if self.sampling_rate:
                    self.sampling_rate = (0.9 * self.sampling_rate) + (0.1 * instant_fs)
                else:
                    self.sampling_rate = instant_fs
                    
                # Update the GUI label to show the live Fs
                # (Optional: Only update text occasionally to avoid flickering)
                if not self.manual_fs_override:
                    self.fs_input.setText(f"{self.sampling_rate:.1f}")

            # Reset timer for next cycle
            self.last_plot_update_time = current_time

            self.update_plots()
            self.number_of_data_lines = 0 # Reset counter
        self.update_save_button_state()

    def disconnect_serial(self, prompt_save=True):
        # (Largely same as original, ensure disconnect from correct read signal)
        if self.fft_mode == "file":
            print("Clearing loaded audio file (disconnect called in file mode).")
            self.audio_file_path = None; self.audio_file_data_ch1 = np.array([]); self.audio_file_data_ch2 = np.array([])
            self.audio_file_sampling_rate = None; self.selected_file_label.setText("No file selected.")
            self.clear_plot_data(prompt_save=False) # Clears data arrays too
            self.sampling_rate = None; self.manual_fs_override = False
            self.fs_input.setText(""); self.fs_input.setReadOnly(True)
            self.status_label.setText("STATUS: Audio file cleared. Select a file or switch mode."); self.status_label.setStyleSheet("color: #0000FF;")
            self.update_fs_input_state(); self.update_save_button_state(); return

        print("Disconnect requested...")
        if prompt_save:
            save_result = self._trigger_save_if_needed(context="disconnect_serial")
            if save_result == "cancel":
                print("Disconnection cancelled by user due to save dialog cancel.")
                self.update_ui_after_stop_or_save(); return
        
        if self.is_recording: self._internal_stop_recording_logic(ask_to_save=False, triggered_by_error=False)

        if self.serial.isOpen():
            try: self.serial.readyRead.disconnect(self.initiating_read_dual_channel)
            except (RuntimeError, TypeError): pass
            try: self.serial.readyRead.disconnect(self.read_data_dual_channel)
            except (RuntimeError, TypeError): pass
            self.serial.close(); print("Serial port closed.")

        if self.csv_temp_file_path and os.path.exists(self.csv_temp_file_path) and not self.prompt_save_on_disconnect_after_stop:
            print(f"Disconnecting: Removing unhandled temp file {self.csv_temp_file_path}")
            try: os.remove(self.csv_temp_file_path)
            except Exception as e: print(f"Error removing temp CSV during disconnect: {e}")
            self.csv_temp_file_path = None
        
        self.is_initiating = False; self.is_ready_to_record = False; self.is_recording = False
        self.continuing_plot_after_stop = False
        self.reset_to_disconnected_state()
        if not self.manual_fs_override : self.fs_input.setText("") # Clear Fs if it was auto-detected
        self.status_label.setText("STATUS: Disconnected")
        self.status_label.setStyleSheet("color: #FF0000;")
        self.update_save_button_state()

    def reset_to_disconnected_state(self):
        # (Largely same as original)
        current_status_text = self.status_label.text()
        if not any(keyword in current_status_text for keyword in ["Saved", "Discarded", "Error", "Reset", "Cleared", "Disconnected", "pending save", "Timeout", "Ready"]):
             self.status_label.setText("STATUS: Disconnected")
             self.status_label.setStyleSheet("color: #FF0000;")

        self.initiate_start_button.setText("Initiate")
        self.disconnect_button.setEnabled(False); self.stop_recording_button.setEnabled(False)

        if self.fft_mode == "realtime":
            self.port_select.setEnabled(True); self.baud_combobox.setEnabled(True); self.refresh_button.setEnabled(True)
            # Check if QSerialPortInfo.availablePorts() returns a non-empty list
            available_ports = QSerialPortInfo.availablePorts()
            self.initiate_start_button.setEnabled(bool(available_ports))
        else: # File mode
            self.port_select.setEnabled(False); self.baud_combobox.setEnabled(False); self.refresh_button.setEnabled(False)
            self.initiate_start_button.setEnabled(False)
        
        self.continuing_plot_after_stop = False
        if not self.manual_fs_override: self.fs_input.setText("")
        self.update_fs_input_state()
        self.update_com_ports() 
        self.update_save_button_state()
        print("UI reset to disconnected state (respecting FFT mode).")

    def _plot_fft(self, data, fs, plot_widget, title_base):
        N = len(data)
        plot_widget.setTitle(title_base) # Set title first
        plot_widget.setLabel("left", "Magnitude")
        plot_widget.setLabel("bottom", "Frequency (Hz)")

        if N <= 1 or fs is None or fs <= 0:
            plot_widget.clear()
            plot_widget.setTitle(title_base + " (No Data/Fs)")
            return

        remove_dc = True
        
        # 2. Use the SignalProcessor to do the heavy math
        freqs, magnitudes = self.signal_processor.compute_fft(data, fs, remove_dc=remove_dc)
        
        if len(freqs) > 0:
            plot_widget.plot(freqs, magnitudes, pen=mkPen('r'), clear=True)
            self.sync_zoom_plot(plot_widget, freqs, magnitudes, 'r')
        else:
            plot_widget.clear()
            plot_widget.setTitle(title_base + " (FFT Error)")

    def _plot_filtered_signal(self, data, fs, plot_widget, title_base):
        N = len(data)
        plot_widget.setTitle(title_base)
        plot_widget.setLabel("left", "Value (Filtered)")
        plot_widget.setLabel("bottom", "Sample Index")

        # 1. Check if we have data
        if N <= 1 or fs is None or fs <= 0:
            plot_widget.clear()
            plot_widget.setTitle(title_base + " (No Data/Fs)")
            return

        # 2. Setup Default (Raw data if filtering fails or is disabled)
        filtered_plot_data = data 
        min_len_for_filter = self.filter_order * 3 + 1 
        have_enough_data_for_filter = N >= min_len_for_filter

        # 3. Apply Filtering if enabled
        if self.sliders_enabled and have_enough_data_for_filter:
            nyquist = fs / 2.0
            
            # Prepare cutoffs
            effective_highcut = min(self.highcut, nyquist * 0.999) if self.highcut > 0 else nyquist * 0.999
            effective_lowcut = self.lowcut

            # Determine which filters to use
            apply_lowpass = effective_highcut > 0 and effective_highcut < nyquist
            apply_highpass = effective_lowcut > 0 and effective_lowcut < nyquist
            
            # Validate range (Low must be < High)
            range_valid = True
            if apply_lowpass and apply_highpass and effective_lowcut >= effective_highcut:
                range_valid = False

            temp_filtered_data = data.copy()
            filtering_applied_this_cycle = False
            filter_error_this_cycle = False

            if not range_valid:
                filter_error_this_cycle = True

            if not filter_error_this_cycle:
                try:
                    # --- Highpass Step ---
                    if apply_highpass:
                        b_hp, a_hp = self.signal_processor.design_highpass(effective_lowcut, fs, order=self.filter_order)
                        if b_hp is not None and a_hp is not None:
                            temp_filtered_data = signal.filtfilt(b_hp, a_hp, temp_filtered_data)
                            filtering_applied_this_cycle = True
                        else: 
                            filter_error_this_cycle = True
                            print(f"Highpass design failed for {title_base}")
                    
                    # --- Lowpass Step ---
                    if apply_lowpass and not filter_error_this_cycle:
                        b_lp, a_lp = self.signal_processor.design_lowpass(effective_highcut, fs, order=self.filter_order)
                        if b_lp is not None and a_lp is not None:
                            temp_filtered_data = signal.filtfilt(b_lp, a_lp, temp_filtered_data)
                            filtering_applied_this_cycle = True
                        else: 
                            filter_error_this_cycle = True
                            print(f"Lowpass design failed for {title_base}")

                    # --- CRITICAL STEP: Update the plot data ---
                    if filtering_applied_this_cycle and not filter_error_this_cycle:
                        filtered_plot_data = temp_filtered_data
                        # Trim start/end artifacts
                        mask_size = self.filter_order * 3
                        if len(filtered_plot_data) > mask_size:
                            filtered_plot_data = filtered_plot_data[:-mask_size]
                        
                except Exception as e:
                    print(f"Unexpected error during filtering for {title_base}: {e}")
        
        # 4. Draw the Plot
        plot_widget.plot(np.arange(len(filtered_plot_data)), filtered_plot_data, pen=mkPen('g'), clear=True)
        self.sync_zoom_plot(plot_widget, np.arange(len(filtered_plot_data)), filtered_plot_data, 'g')

    def update_plots(self):
        # Determine data source and type based on mode
        if self.fft_mode == "file":
            if self.audio_file_sampling_rate is None: # No file loaded or error
                self.clear_all_plot_widgets(); return

            data_ch1_to_plot = self.audio_file_data_ch1
            data_ch2_to_plot = self.audio_file_data_ch2
            current_fs = self.audio_file_sampling_rate
            
            raw_title_base = "Raw Audio Ch"
            filt_title_base = "Filtered Audio Ch"
            fft_title_base = "FFT Audio Ch"
            y_label_raw = "Amplitude"

        elif self.fft_mode == "realtime":
            if not (self.is_recording or self.continuing_plot_after_stop or (self.is_ready_to_record and self.sampling_rate)):
                self.clear_all_plot_widgets(); return # Not ready to plot realtime
            if not self.sampling_rate or self.sampling_rate <= 0:
                self.status_label.setText("STATUS: Fs Invalid for Plotting!"); self.status_label.setStyleSheet("color: #FF0000;")
                self.clear_all_plot_widgets(); return

            data_ch1_to_plot = self.data_array_raw_ch1
            data_ch2_to_plot = self.data_array_raw_ch2
            current_fs = self.sampling_rate # Rate of RMS value pairs

            raw_title_base = "RMS Ch"
            filt_title_base = "Filtered RMS Ch"
            fft_title_base = "FFT of RMS Ch" # FFT of the RMS envelope
            y_label_raw = "RMS Value"
        else:
            self.clear_all_plot_widgets(); return # Should not happen

        # --- Plot Channel 1 ---
        N1 = len(data_ch1_to_plot)
        if N1 > 1:
            self.raw_signal_plot_ch1.setTitle(f"{raw_title_base}1")
            self.raw_signal_plot_ch1.setLabel("left", y_label_raw)
            # For realtime RMS, data is likely 0-1. For audio, it's normalized -1 to 1.
            self.raw_signal_plot_ch1.plot(np.arange(N1), data_ch1_to_plot, pen=mkPen('b'), clear=True)
            self.sync_zoom_plot(self.raw_signal_plot_ch1, np.arange(N1), data_ch1_to_plot, 'b')
            self._plot_fft(data_ch1_to_plot, current_fs, self.fft_plot_ch1, f"{fft_title_base}1")
            self._plot_filtered_signal(data_ch1_to_plot, current_fs, self.filtered_signal_plot_ch1, f"{filt_title_base}1")
        else:
            self.raw_signal_plot_ch1.clear(); self.fft_plot_ch1.clear(); self.filtered_signal_plot_ch1.clear()
            self.raw_signal_plot_ch1.setTitle(f"{raw_title_base}1 (No Data)")
            self.fft_plot_ch1.setTitle(f"{fft_title_base}1 (No Data)")
            self.filtered_signal_plot_ch1.setTitle(f"{filt_title_base}1 (No Data)")


        # --- Plot Channel 2 ---
        N2 = len(data_ch2_to_plot)
        if N2 > 1:
            self.raw_signal_plot_ch2.setTitle(f"{raw_title_base}2")
            self.raw_signal_plot_ch2.setLabel("left", y_label_raw)
            self.raw_signal_plot_ch2.plot(np.arange(N2), data_ch2_to_plot, pen=mkPen('c'), clear=True) # Cyan for Ch2
            self.sync_zoom_plot(self.raw_signal_plot_ch2, np.arange(N2), data_ch2_to_plot, 'c')
            self._plot_fft(data_ch2_to_plot, current_fs, self.fft_plot_ch2, f"{fft_title_base}2")
            self._plot_filtered_signal(data_ch2_to_plot, current_fs, self.filtered_signal_plot_ch2, f"{filt_title_base}2")
        elif self.fft_mode == "file" and N1 > 0 and N2 == 0: # Mono file, Ch2 plots show "Mono"
            self.raw_signal_plot_ch2.clear(); self.fft_plot_ch2.clear(); self.filtered_signal_plot_ch2.clear()
            self.raw_signal_plot_ch2.setTitle(f"{raw_title_base}2 (Mono File)")
            self.fft_plot_ch2.setTitle(f"{fft_title_base}2 (Mono File)")
            self.filtered_signal_plot_ch2.setTitle(f"{filt_title_base}2 (Mono File)")
        else: # No data for Ch2
            self.raw_signal_plot_ch2.clear(); self.fft_plot_ch2.clear(); self.filtered_signal_plot_ch2.clear()
            self.raw_signal_plot_ch2.setTitle(f"{raw_title_base}2 (No Data)")
            self.fft_plot_ch2.setTitle(f"{fft_title_base}2 (No Data)")
            self.filtered_signal_plot_ch2.setTitle(f"{filt_title_base}2 (No Data)")
        
        # Update status label for filtering if in real-time recording
        if self.fft_mode == "realtime" and self.is_recording:
            current_status = self.status_label.text()
            status_fs_text = f"(Manual Fs: {current_fs:.1f} Hz)" if self.manual_fs_override else f"(Fs: {current_fs:.1f} Hz)"
            if self.sliders_enabled and not any(s in current_status for s in ["Error", "Warning", "Timeout", "Stopped", "Pending"]):
                 self.status_label.setText(f"STATUS: Recording & Filtering {status_fs_text}"); self.status_label.setStyleSheet("color: #00DD00;")
            elif not self.sliders_enabled and not any(s in current_status for s in ["Error", "Warning", "Timeout", "Stopped", "Pending"]):
                 self.status_label.setText(f"STATUS: Recording {status_fs_text}"); self.status_label.setStyleSheet("color: #00DD00;")


    def clear_all_plot_widgets(self):
        self.raw_signal_plot_ch1.clear(); self.raw_signal_plot_ch2.clear()
        self.filtered_signal_plot_ch1.clear(); self.filtered_signal_plot_ch2.clear()
        self.fft_plot_ch1.clear(); self.fft_plot_ch2.clear()
        # Set titles to indicate no data
        self.raw_signal_plot_ch1.setTitle("Channel 1 (Raw - No Data)")
        self.raw_signal_plot_ch2.setTitle("Channel 2 (Raw - No Data)")
        self.filtered_signal_plot_ch1.setTitle("Channel 1 (Filtered - No Data)")
        self.filtered_signal_plot_ch2.setTitle("Channel 2 (Filtered - No Data)")
        self.fft_plot_ch1.setTitle("Channel 1 (FFT - No Data)")
        self.fft_plot_ch2.setTitle("Channel 2 (FFT - No Data)")


    def clear_plot_data(self, prompt_save=True):
        print("Clearing plot data and visual plots for dual channels.")
        if prompt_save:
            save_result = self._trigger_save_if_needed(context="clear_plots")
            if save_result == "cancel":
                print("Clear plots cancelled by user during save prompt.")
                self.update_ui_after_stop_or_save(); return

        # Clear data arrays for both channels
        self.data_array_raw_ch1 = np.array([])
        self.data_array_raw_ch2 = np.array([])
        # self.data_array_fined_ch1 = np.array([]) # If used
        # self.data_array_fined_ch2 = np.array([]) # If used

        if self.fft_mode == "file": # Also clear audio file specific data
            self.audio_file_path = None
            self.audio_file_data_ch1 = np.array([])
            self.audio_file_data_ch2 = np.array([])
            self.audio_file_sampling_rate = None
            self.selected_file_label.setText("No file selected.")
            # self.sampling_rate = None # Keep Fs if manually set for realtime, or from file if file mode
            # self.manual_fs_override = False
            # self.fs_input.setText("")
            # self.fs_input.setReadOnly(True) # Re-evaluate this based on mode
            if not self.is_recording: # Don't overwrite recording status
                self.status_label.setText("STATUS: Audio file data cleared.")
                self.status_label.setStyleSheet("color: #0000FF;")


        self.number_of_data_lines = 0
        self.clear_all_plot_widgets() # Clear the plot widgets themselves

        self.continuing_plot_after_stop = False
        self.update_save_button_state()
        # If in file mode and plots were cleared, ensure Fs input reflects that no file is loaded
        if self.fft_mode == "file" and self.audio_file_sampling_rate is None:
            self.fs_input.setText("")
            self.fs_input.setReadOnly(True)


    def reset_alldata(self):
        print("Resetting all data and connection for dual channels...")
        save_result = self._trigger_save_if_needed(context="reset_all")
        if save_result == "cancel":
            print("Reset All cancelled by user during save prompt."); return

        if self.is_recording: self._internal_stop_recording_logic(ask_to_save=False, triggered_by_error=False)
        if self.serial.isOpen():
            try: self.serial.readyRead.disconnect(self.initiating_read_dual_channel)
            except (RuntimeError, TypeError): pass
            try: self.serial.readyRead.disconnect(self.read_data_dual_channel)
            except (RuntimeError, TypeError): pass
            self.serial.close(); print("Serial port closed by Reset All.")

        self.close_csv_file()
        if self.csv_temp_file_path and os.path.exists(self.csv_temp_file_path):
            print(f"Reset All: Removing temp file {self.csv_temp_file_path}")
            try: os.remove(self.csv_temp_file_path)
            except Exception as e: print(f"Error removing temp CSV during reset: {e}")
        self.csv_temp_file_path = None
        self.prompt_save_on_disconnect_after_stop = False
        
        is_realtime_checked_before_reset = self.realtime_fft_radio.isChecked()
        
        self.init_variables() # Re-initialize all variables (includes dual channel arrays)
        
        # Restore FFT mode
        if is_realtime_checked_before_reset:
            self.fft_mode = "realtime"
            self.realtime_fft_radio.setChecked(True)
        else:
            self.fft_mode = "file"
            self.file_fft_radio.setChecked(True)

        self.clear_plot_data(prompt_save=False) # Clears arrays and plots

        self.reset_sliders()
        self.toggle_button.setChecked(False) # Default to filtering off
        self.toggle_sliders() # Apply the state

        default_baud_item = "921600"
        index = self.baud_combobox.findText(default_baud_item)
        if index != -1: self.baud_combobox.setCurrentIndex(index)
        else: self.baud_combobox.setCurrentIndex(0) if self.baud_combobox.count() > 0 else None
        
        self.handle_fft_mode_change() # This will set up UI based on mode

        self.status_label.setText("STATUS: Reset Complete. Disconnected.")
        self.status_label.setStyleSheet("color: #FF0000;")
        self.update_save_button_state()
        print("Dual Channel Reset complete.")

    def handle_stop_recording_button(self, triggered_by_error=False): # Added for completeness
        if self.fft_mode == "file": return
        print("Stop Recording button clicked.")
        self._internal_stop_recording_logic(ask_to_save=True, triggered_by_error=triggered_by_error)

    def _internal_stop_recording_logic(self, ask_to_save=True, triggered_by_error=False): # Added for completeness
        was_actively_recording = self.is_recording
        self.is_recording = False
        if self.serial.isOpen():
            try: self.serial.readyRead.disconnect(self.read_data_dual_channel) # Disconnect new method
            except (RuntimeError, TypeError): pass
        
        self.flush_csv_buffer()
        self.close_csv_file()
        data_file_exists = self.csv_temp_file_path and os.path.exists(self.csv_temp_file_path)

        if triggered_by_error:
            if data_file_exists:
                print(f"Recording stopped due to error. Discarding temp file: {self.csv_temp_file_path}")
                try: os.remove(self.csv_temp_file_path)
                except Exception as e: print(f"Error removing temp CSV after error: {e}")
                self.csv_temp_file_path = None
                self.prompt_save_on_disconnect_after_stop = False
        elif ask_to_save:
            if (was_actively_recording or self.prompt_save_on_disconnect_after_stop) and data_file_exists:
                current_context = "stop_recording_button" if was_actively_recording else "pending_save_retriggered"
                self.show_save_dialog(context=current_context)
        self.update_ui_after_stop_or_save()

    def update_ui_after_stop_or_save(self): # Added for completeness
        fs_text = f"(Fs: {self.sampling_rate:.1f} Hz)" if self.sampling_rate else ""
        fs_text = f"(Manual Fs: {self.sampling_rate:.1f} Hz)" if self.manual_fs_override and self.sampling_rate else fs_text

        if self.prompt_save_on_disconnect_after_stop:
            self.status_label.setText(f"STATUS: Recording Stopped. Data pending save. {fs_text}")
            self.status_label.setStyleSheet("color: #FFA500;")
        elif self.csv_temp_file_path is None and not self.is_recording: # Data clear (saved/discarded)
             self.status_label.setText(f"STATUS: Data cleared. Ready to Start or Disconnect. {fs_text}")
             self.status_label.setStyleSheet("color: #00A0A0;")
             # Do not clear plots here automatically, user might want to see last data
        else: # Stopped, but temp file might still exist if save was cancelled earlier
            self.status_label.setText(f"STATUS: Recording Stopped. Ready to Start or Disconnect. {fs_text}")
            self.status_label.setStyleSheet("color: #FFA500;")

        if self.sampling_rate is not None and self.serial.isOpen() and self.fft_mode == "realtime":
            self.is_ready_to_record = True
            # Check if there's any data to continue plotting from
            has_data_ch1 = self.data_array_raw_ch1.size > 0
            has_data_ch2 = self.data_array_raw_ch2.size > 0
            self.continuing_plot_after_stop = has_data_ch1 or has_data_ch2

            self.initiate_start_button.setText("Start Recording")
            self.initiate_start_button.setEnabled(True)
        elif self.fft_mode == "realtime": # Not ready or serial closed
            self.is_ready_to_record = False
            self.continuing_plot_after_stop = False
            if not self.serial.isOpen(): self.reset_to_disconnected_state()
            else: # Serial open, but Fs might be missing
                 self.initiate_start_button.setText("Initiate")
                 self.initiate_start_button.setEnabled(True) # Can always try to re-initiate
        
        self.stop_recording_button.setEnabled(False) # Recording is now stopped
        self.disconnect_button.setEnabled(self.serial.isOpen() or self.fft_mode == "file") # Can disconnect if serial open or clear file
        self.update_fs_input_state()
        self.update_save_button_state()

    def update_fs_input_state(self): # Added for completeness
        if self.fft_mode == "realtime":
            can_set_fs = (self.is_ready_to_record or self.is_recording) and self.serial.isOpen()
            self.fs_input.setEnabled(can_set_fs)
            self.fs_input.setReadOnly(not can_set_fs)
            self.apply_fs_button.setEnabled(can_set_fs)
            if not can_set_fs and not self.is_initiating and not self.manual_fs_override:
                 self.fs_input.setText("")
        elif self.fft_mode == "file":
            self.fs_input.setEnabled(True) # Show Fs from file
            self.fs_input.setReadOnly(True) # But don't allow edit
            self.apply_fs_button.setEnabled(False)
            if self.audio_file_sampling_rate:
                self.fs_input.setText(f"{self.audio_file_sampling_rate:.1f}")
            else:
                self.fs_input.setText("")
        
        if self.sampling_rate is None and not self.is_initiating and self.fft_mode == "realtime":
            if not self.manual_fs_override : self.fs_input.setText("")

    def handle_apply_fs_button(self): # Added for completeness
        if self.fft_mode != "realtime": return
        try:
            manual_fs = float(self.fs_input.text())
            if manual_fs >= MIN_MANUAL_FS_INPUT:
                self.sampling_rate = manual_fs; self.manual_fs_override = True
                print(f"Manual Fs applied: {self.sampling_rate} Hz (lines/s)")
                status_text_core = f"Manual Fs: {self.sampling_rate:.1f} Hz"
                if self.is_recording:
                    self.status_label.setText(f"STATUS: Recording ({status_text_core})"); self.status_label.setStyleSheet("color: #00DD00;")
                elif self.is_ready_to_record:
                    self.status_label.setText(f"STATUS: Ready ({status_text_core})"); self.status_label.setStyleSheet("color: #00FF00;")
                else: # Not recording, not initiated fully, but Fs set.
                    self.status_label.setText(f"STATUS: {status_text_core} - Ready to Start/Initiate"); self.status_label.setStyleSheet("color: #00FF00;")
                
                if self.sliders_enabled: # Re-evaluate filters with new Fs
                    self.reset_sliders() 
                    self.on_text1_changed(self.number_input1.text())
                    self.on_text2_changed(self.number_input2.text())
                self.update_plots() # Update plots with new Fs if data exists
            else:
                QMessageBox.warning(self, "Invalid Fs", f"Sampling frequency must be a positive number (>= {MIN_MANUAL_FS_INPUT}).")
                # Restore previous Fs text if invalid input
                if self.sampling_rate is not None: self.fs_input.setText(f"{self.sampling_rate:.1f}")
                else: self.fs_input.setText("")
        except ValueError:
            QMessageBox.warning(self, "Invalid Fs", "Please enter a valid number for Fs.")
            if self.sampling_rate is not None: self.fs_input.setText(f"{self.sampling_rate:.1f}")
            else: self.fs_input.setText("")
        self.update_save_button_state()

    def on_text1_changed(self, text): # Filter Lowcut (largely original logic)
        current_lowcut = self.lowcut
        try:
            value = float(text)
            valid_input = False
            max_freq = MAX_MANUAL_FREQ_INPUT # Absolute max
            # Effective max depends on Nyquist from current Fs
            current_fs = self.sampling_rate if self.sampling_rate else MAX_MANUAL_FREQ_INPUT * 2 
            effective_max_cutoff = min(max_freq, (current_fs / 2.0) * 0.99) if current_fs > 0 else max_freq

            if self.sliders_enabled: # If filters active, lowcut must be < highcut
                if 0 <= value < self.highcut and value <= effective_max_cutoff:
                    self.lowcut = value; valid_input = True
            else: # If filters disabled, just validate range
                if 0 <= value <= effective_max_cutoff:
                    self.lowcut = value; valid_input = True
            
            if not valid_input:
                self.lowcut = current_lowcut # Revert
                if text and text!="-": # Avoid clearing if user is typing negative or empty
                    self.number_input1.blockSignals(True)
                    self.number_input1.setText(str(int(self.lowcut)))
                    self.number_input1.blockSignals(False)
        except ValueError: # Non-numeric input
            self.lowcut = current_lowcut
            if text and text!="-":
                self.number_input1.blockSignals(True)
                self.number_input1.setText(str(int(self.lowcut)))
                self.number_input1.blockSignals(False)
        finally:
            self.Min_value_label.setText(f"Low Cut: {self.lowcut:.0f} Hz")
            # Update plots if filters enabled and data is present
            if self.sliders_enabled and (self.is_recording or self.continuing_plot_after_stop or self.fft_mode == "file"):
                self.update_plots()

    def on_text2_changed(self, text): # Filter Highcut (largely original logic)
        current_highcut = self.highcut
        try:
            value = float(text)
            valid_input = False
            max_freq = MAX_MANUAL_FREQ_INPUT
            current_fs = self.sampling_rate if self.sampling_rate else MAX_MANUAL_FREQ_INPUT * 2
            effective_max_cutoff = min(max_freq, (current_fs / 2.0) * 0.99) if current_fs > 0 else max_freq

            if self.sliders_enabled: # Highcut must be > lowcut
                if value > self.lowcut and 0 < value <= effective_max_cutoff : # Highcut must be > 0
                    self.highcut = value; valid_input = True
            else: # Filters disabled, just validate range
                if 0 < value <= effective_max_cutoff:
                    self.highcut = value; valid_input = True

            if not valid_input:
                self.highcut = current_highcut
                if text and text!="-":
                    self.number_input2.blockSignals(True)
                    self.number_input2.setText(str(int(self.highcut)))
                    self.number_input2.blockSignals(False)
        except ValueError:
            self.highcut = current_highcut
            if text and text!="-":
                self.number_input2.blockSignals(True)
                self.number_input2.setText(str(int(self.highcut)))
                self.number_input2.blockSignals(False)
        finally:
            self.Max_value_label.setText(f"High Cut: {self.highcut:.0f} Hz")
            if self.sliders_enabled and (self.is_recording or self.continuing_plot_after_stop or self.fft_mode == "file"):
                self.update_plots()

    def reset_sliders(self): # (Largely original)
        default_low = 0.0
        default_high = DEFAULT_HIGHCUT
        if self.sampling_rate and self.sampling_rate > 0:
             nyquist = self.sampling_rate / 2.0
             default_high = min(DEFAULT_HIGHCUT, nyquist * 0.99) # Ensure default highcut is valid
        
        self.lowcut = default_low
        self.highcut = default_high if default_high > default_low else DEFAULT_HIGHCUT # Ensure high > low

        self.number_input1.blockSignals(True); self.number_input2.blockSignals(True)
        self.number_input1.setText(str(int(self.lowcut)))
        self.number_input2.setText(str(int(self.highcut)))
        self.number_input1.blockSignals(False); self.number_input2.blockSignals(False)

        self.Min_value_label.setText(f"Low Cut: {self.lowcut:.0f} Hz")
        self.Max_value_label.setText(f"High Cut: {self.highcut:.0f} Hz")
        print("Filter inputs reset to defaults based on current Fs.")
        if self.sliders_enabled and (self.is_recording or self.continuing_plot_after_stop or self.fft_mode == "file"):
            self.update_plots()

    def toggle_sliders(self): # (Largely original)
        self.sliders_enabled = self.toggle_button.isChecked()
        is_enabled = self.sliders_enabled
        self.number_input1.setEnabled(is_enabled)
        self.number_input2.setEnabled(is_enabled)
        self.reset_filters_button.setEnabled(is_enabled)
        if is_enabled:
            self.toggle_button.setText("Disable Filtering")
            # Ensure current text values are parsed and applied
            self.on_text1_changed(self.number_input1.text()) 
            self.on_text2_changed(self.number_input2.text())
            print("Filtering Controls Enabled")
        else:
            self.toggle_button.setText("Enable Filtering")
            print("Filtering Controls Disabled")
        # Update plots if data is present, as enabling/disabling filters changes display
        if self.is_recording or self.continuing_plot_after_stop or self.fft_mode == "file":
             self.update_plots()

    # def butter_lowpass(self, cutoff, fs, order=5): # (Original)
    #     if fs is None or fs <= 0: return None, None
    #     nyq = 0.5 * fs; 
    #     normal_cutoff = min(cutoff, nyq * 0.999) / nyq # Ensure cutoff is less than Nyquist
    #     if normal_cutoff <=0: return None, None # Cutoff must be positive
    #     try: return signal.butter(order, normal_cutoff, btype='low', analog=False)
    #     except ValueError as e: print(f"Err creating lowpass butter filter (cutoff:{cutoff}, fs:{fs}): {e}"); return None, None

    # def butter_highpass(self, cutoff, fs, order=5): # (Original)
    #     if fs is None or fs <= 0: return None, None
    #     nyq = 0.5 * fs; 
    #     normal_cutoff = min(cutoff, nyq * 0.999) / nyq
    #     if normal_cutoff <=0: return None, None
    #     try: return signal.butter(order, normal_cutoff, btype='high', analog=False)
    #     except ValueError as e: print(f"Err creating highpass butter filter (cutoff:{cutoff}, fs:{fs}): {e}"); return None, None

    def handle_save_data_button(self): # (Largely original)
        print("Save Data button clicked.")
        if not self._check_if_save_needed():
            QMessageBox.information(self, "No Data", "No temporary data to save or data has been handled.")
            return
        was_recording_when_clicked = self.is_recording
        save_result = self.show_save_dialog(context="manual_save")
        if save_result == "saved" or save_result == "discarded":
            self.prompt_save_on_disconnect_after_stop = False
            if was_recording_when_clicked: # If it was recording, it's now stopped by save dialog
                self.is_recording = False 
                if self.serial.isOpen():
                    try: self.serial.readyRead.disconnect(self.read_data_dual_channel)
                    except (RuntimeError, TypeError) as e: print(f"Warning: disconnect read_data in save_button: {e}")
            self.update_ui_after_stop_or_save()
        elif save_result == "cancel":
            print("Save operation cancelled by user.")
            if self.csv_temp_file_path and os.path.exists(self.csv_temp_file_path):
                 self.prompt_save_on_disconnect_after_stop = True # Mark as pending if file still exists
            else: self.prompt_save_on_disconnect_after_stop = False
            if was_recording_when_clicked: self.is_recording = False # Stop recording, data is pending
            self.update_ui_after_stop_or_save()

    def show_save_dialog(self, context="unknown"): # (Largely original)
        if not self.csv_temp_file_path or not os.path.exists(self.csv_temp_file_path):
            print(f"show_save_dialog (context: {context}): No temp file to save or path invalid.")
            self.csv_temp_file_path = None; self.prompt_save_on_disconnect_after_stop = False
            return "no_file"
        self.flush_csv_buffer(); self.close_csv_file() # Ensure all data written and file closed before moving/deleting
        msg_box = QMessageBox(self); msg_box.setIcon(QMessageBox.Question)
        msg_box.setWindowTitle("Save Recording?"); msg_box.setText(f"Recorded data exists:\n{os.path.basename(self.csv_temp_file_path)}\n\nSave it permanently?")
        msg_box.setStandardButtons(QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel)
        msg_box.setDefaultButton(QMessageBox.Save); ret = msg_box.exec()
        action_taken = "cancel"
        if ret == QMessageBox.Save:
            default_filename = f"rec_dual_{time.strftime('%Y%m%d_%H%M%S')}.csv"
            save_path, _ = QFileDialog.getSaveFileName(self, "Save As", default_filename, "CSV (*.csv)")
            if save_path:
                try:
                    os.replace(self.csv_temp_file_path, save_path) 
                    self.status_label.setText(f"STATUS: Saved to {os.path.basename(save_path)}"); self.status_label.setStyleSheet("color: #00A000;")
                    print(f"Data saved from {self.csv_temp_file_path} to: {save_path}")
                    self.csv_temp_file_path = None; action_taken = "saved"
                except Exception as e:
                    QMessageBox.critical(self, "Save Error", f"Could not save file:\n{e}")
                    self.status_label.setText("STATUS: Save Error!"); self.status_label.setStyleSheet("color: #FF8C00;")
                    action_taken = "error_saving" # Temp file still exists, user can retry
            else: self.status_label.setText("STATUS: Save cancelled by user."); self.status_label.setStyleSheet("color: #FFA500;"); action_taken = "cancel"
        elif ret == QMessageBox.Discard:
            try:
                os.remove(self.csv_temp_file_path)
                self.status_label.setText("STATUS: Data discarded."); self.status_label.setStyleSheet("color: #FFA500;")
                print(f"Temporary data discarded: {self.csv_temp_file_path}")
                self.csv_temp_file_path = None; action_taken = "discarded"
            except Exception as e:
                QMessageBox.critical(self, "Discard Error", f"Could not discard temporary file:\n{e}")
                self.status_label.setText("STATUS: Discard Error!"); self.status_label.setStyleSheet("color: #FF8C00;")
                action_taken = "error_discarding" # Temp file still exists
        elif ret == QMessageBox.Cancel:
            self.status_label.setText("STATUS: Save operation cancelled."); self.status_label.setStyleSheet("color: #FFA500;"); action_taken = "cancel"
        
        if action_taken in ["saved", "discarded"]: self.prompt_save_on_disconnect_after_stop = False
        elif self.csv_temp_file_path and os.path.exists(self.csv_temp_file_path): self.prompt_save_on_disconnect_after_stop = True
        else: self.prompt_save_on_disconnect_after_stop = False; self.csv_temp_file_path = None
        return action_taken

    def flush_csv_buffer(self): # (Largely original)
        if self.csv_writer and self.csv_buffer and self.csv_file and not self.csv_file.closed:
            try:
                self.csv_writer.writerows(self.csv_buffer); self.csv_buffer = []; self.csv_file.flush()
            except Exception as e:
                print(f"Err flushing CSV: {e}")
                if self.is_recording: 
                    self._internal_stop_recording_logic(ask_to_save=True, triggered_by_error=True)
                    self.status_label.setText("STATUS: CSV Flush Error! Rec Stopped."); self.status_label.setStyleSheet("color: #FFA500;")

    def close_csv_file(self): # (Largely original)
        if self.csv_file and not self.csv_file.closed:
            try: self.csv_file.close()
            except Exception as e: print(f"Err closing CSV: {e}")
        self.csv_file = None; self.csv_writer = None

    def _check_if_save_needed(self): # (Largely original)
        data_exists = bool(self.csv_temp_file_path and os.path.exists(self.csv_temp_file_path))
        if not data_exists and self.prompt_save_on_disconnect_after_stop:
            print("Warning: prompt_save_on_disconnect_after_stop was true, but no temp CSV file found. Resetting flag.")
            self.prompt_save_on_disconnect_after_stop = False
        return data_exists

    def _trigger_save_if_needed(self, context="unknown"): # (Largely original)
        if self._check_if_save_needed(): return self.show_save_dialog(context=context)
        return "no_prompt_needed"

    def update_save_button_state(self): # (Largely original)
        can_save_csv = self._check_if_save_needed()
        self.save_data_button.setEnabled(can_save_csv)
        if self.fft_mode == "realtime" and can_save_csv and self.prompt_save_on_disconnect_after_stop:
            current_status_text = self.status_label.text()
            protected_statuses_keywords = ["Recording", "Initiating", "Error", "Timeout", "Saving", "Saved", "Discarded", "pending save"]
            is_protected_status = any(keyword in current_status_text for keyword in protected_statuses_keywords)
            if not is_protected_status:
                new_status_message = "STATUS: Unsaved recorded data (pending user action)."
                if current_status_text != new_status_message:
                    self.status_label.setText(new_status_message); self.status_label.setStyleSheet("color: #DAA520;")

    def closeEvent(self, event): 
        print("Close event triggered.")
        save_result = self._trigger_save_if_needed(context="window_close")
        if save_result == "cancel":
            print("Window close cancelled by user during save prompt."); event.ignore(); return
        if self.serial.isOpen(): print("Closing serial port on exit..."); self.serial.close()
        if self.csv_temp_file_path and os.path.exists(self.csv_temp_file_path):
            print(f"Removing unhandled/unsaved temp file on exit: {self.csv_temp_file_path}")
            try: os.remove(self.csv_temp_file_path)
            except Exception as e: print(f"Error removing temp file on application close: {e}")
        event.accept()
    def open_zoom_window(self, source_widget):
        """Opens the popup window for the double-clicked plot."""
        # Close existing zoom window if open
        if self.current_zoom_window:
            self.current_zoom_window.close()
        
        # Get properties from the clicked widget
        title = source_widget.plotItem.titleLabel.text
        x_label = source_widget.plotItem.getAxis('bottom').labelText
        y_label = source_widget.plotItem.getAxis('left').labelText
        
        # Create and show the dialog
        self.current_zoom_window = ZoomDialog(title, x_label, y_label, self)
        self.zoomed_plot_source = source_widget
        self.current_zoom_window.show()
        
        # Force an immediate update so it's not empty
        self.update_plots()
    def sync_zoom_plot(self, source_widget, x, y, color):
        """Sends data to the zoom window if it matches the source widget."""
        if (self.current_zoom_window and 
            self.current_zoom_window.isVisible() and 
            self.zoomed_plot_source == source_widget):
            
            self.current_zoom_window.update_data(x, y, color)
