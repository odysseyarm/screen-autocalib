import sys
import pyrealsense2 as rs
from PySide6.QtWidgets import QApplication, QMainWindow, QStackedWidget
from page1 import Page1
from page2 import Page2
from page3 import Page3
import argparse

class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()

        self.stacked_widget = QStackedWidget()
        self.setCentralWidget(self.stacked_widget)

        self.pipeline: Optional[rs.pipeline] = None

        # Create instances of pages
        self.page1 = Page1(self, self.init_pipeline, self.exit_application)
        self.page2 = Page2(self, self.goto_page3, self.exit_application)
        self.page3 = Page3(self, self.goto_page2, self.exit_application, self.pipeline)

        self.stacked_widget.addWidget(self.page1)
        self.stacked_widget.addWidget(self.page2)
        self.stacked_widget.addWidget(self.page3)

        self.stacked_widget.setCurrentWidget(self.page1)
        self.showFullScreen()

    def init_pipeline(self) -> None:
        # Initialize the RealSense pipeline
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        self.pipeline_profile = self.pipeline.start(config)

        # Create and configure a temporal filter
        self.temporal_filter = rs.temporal_filter()
        self.align = rs.align(rs.stream.color)

        # Pass the pipeline and filter to Page3
        self.page3.pipeline = self.pipeline
        self.page3.temporal_filter = self.temporal_filter
        self.page3.align = self.align

        self.goto_page2()

    def goto_page3(self) -> None:
        self.stacked_widget.setCurrentWidget(self.page3)

    def goto_page2(self) -> None:
        self.stacked_widget.setCurrentWidget(self.page2)

    def exit_application(self) -> None:
        if hasattr(self, 'pipeline') and self.pipeline:
            self.pipeline.stop()
        sys.exit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RealSense Camera GUI')
    parser.add_argument('--display', type=int, default=0, help='Display index to use')
    args = parser.parse_args()

    app = QApplication(sys.argv)

    screens = QApplication.screens()
    if args.display < 0 or args.display >= len(screens):
        print(f"Display index {args.display} is out of range. Using default display 0.")
        args.display = 0
    screen = screens[args.display]

    window = MainWindow()
    window.setGeometry(screen.availableGeometry())
    window.showFullScreen()
    window.windowHandle().setScreen(screen)
    sys.exit(app.exec())
