import sys
from PySide6.QtWidgets import QApplication, QDialog

from class_side import MainWindow
from startup import RulesDialog


if __name__ == "__main__":
    app = QApplication(sys.argv)
    rules_dialog = RulesDialog()
    dialog_result = rules_dialog.exec()
    if dialog_result == QDialog.Accepted:
        window = MainWindow()
        window.showMaximized()
        sys.exit(app.exec())
    else:
        print("Startup information not acknowledged. Exiting.")
        sys.exit(0)
