from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                               QPushButton, QComboBox, QMainWindow,
                               QGridLayout, QApplication, QLineEdit,
                               QMessageBox, QFileDialog, QGroupBox, QRadioButton, QDialog)
from PySide6.QtSerialPort import QSerialPort, QSerialPortInfo
from PySide6.QtCore import QIODevice, Qt
class RulesDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Rules and Announcements")
        # Center the dialog on the screen
        screen = QApplication.primaryScreen()
        if screen: # Check if a primary screen exists
            screen_geometry = screen.availableGeometry()
            dialog_width = 600
            dialog_height = 400
            x = (screen_geometry.width() - dialog_width) // 2
            y = (screen_geometry.height() - dialog_height) // 2
            self.setGeometry(x, y, dialog_width, dialog_height)
        self.setModal(True)

        layout = QVBoxLayout(self)
        rules_text = (
            "Welcome to the Dual Channel Sound Signal Processor!\n\n"
            "Created by TriTan (BMEk23)\n"
            "This application processes sound signals (real-time RMS or offline .wav).\n\n"
            "Please read the following:\n\n"
            "1. Real-time Mode: Expects comma-separated values (e.g., RMS_Ch1,RMS_Ch2)\n"
            "   from the serial port. Ensure your device (e.g., ESP32) is sending data in this format.\n\n"
            "2. File Mode: Supports mono or stereo .wav files. Stereo files will display\n"
            "   both channels.\n\n"
            "3. Recording: Data is temporarily stored. Use 'Save Data' to keep recordings.\n\n"
            "4. Sampling Rate (Fs): For real-time, this is the rate at which data pairs\n"
            "   are received. For files, it's the audio sampling rate.\n\n"
            "5. Filtering/FFT in Real-time: Operations are performed on the incoming RMS values.\n"
            "   The FFT will show the frequency content of the RMS envelope.\n\n"
            "Click 'OK' to acknowledge and continue."
        )
        self.rules_label = QLabel(rules_text)
        self.rules_label.setWordWrap(True)
        layout.addWidget(self.rules_label)

        self.ok_button = QPushButton("OK")
        self.ok_button.clicked.connect(self.accept)
        layout.addWidget(self.ok_button)

        self.setMinimumWidth(500)
        self.setMinimumHeight(350)
