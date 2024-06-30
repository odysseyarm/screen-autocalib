import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from PySide6.QtCore import Qt

from page_1 import Page1
from page_2 import Page2

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Step-by-Step GUI")
        self.setWindowState(Qt.WindowFullScreen)
        self.current_page = None

        self.main_layout = QVBoxLayout()
        self.main_layout.setContentsMargins(0, 0, 0, 0)

        self.container = QWidget()
        self.container.setLayout(self.main_layout)
        self.setCentralWidget(self.container)

        self.show_page(Page1)

    def show_page(self, page_class):
        if self.current_page is not None:
            self.main_layout.removeWidget(self.current_page)
            self.current_page.deleteLater()

        self.current_page = page_class(self)
        self.main_layout.insertWidget(0, self.current_page)

    def next_page(self):
        if isinstance(self.current_page, Page1):
            self.show_page(Page2)
        elif isinstance(self.current_page, Page2):
            None

    def exit_application(self):
        self.close()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
