import sys
import pyrealsense2 as rs
from PySide6.QtWidgets import QApplication, QMainWindow, QStackedWidget, QMessageBox
from page1 import Page1
from page2 import Page2
from page3 import Page3
import argparse
from typing import Optional

class MainWindow(QMainWindow):
    def __init__(self, args: argparse.Namespace) -> None:
        if args.bag:
            self.bag_file = args.bag
        else:
            self.bag_file = None

        super().__init__()

        self.stacked_widget = QStackedWidget()
        self.setCentralWidget(self.stacked_widget)

        self.pipeline: Optional[rs.pipeline] = None

        # Create instances of pages
        self.page2 = Page2(self, self.goto_page3, self.exit_application)
        self.page3 = Page3(self, self.exit_application, self.pipeline, args.screen)

        self.stacked_widget.addWidget(self.page2)
        self.stacked_widget.addWidget(self.page3)

        if self.bag_file is None:
            self.page1 = Page1(self, self.init_pipeline, self.exit_application)
            self.stacked_widget.addWidget(self.page1)
            self.stacked_widget.setCurrentWidget(self.page1)
        else:
            self.init_pipeline()

        self.showFullScreen()

    def init_pipeline(self) -> None:
        # Initialize the RealSense pipeline
        self.pipeline = rs.pipeline()
        config = rs.config()

        try:
            if self.bag_file:
                config.enable_device_from_file(self.bag_file)
                print(f"Streaming from bag file: {self.bag_file}")
            else:
                # Check if any RealSense devices are connected
                context = rs.context()
                if len(context.query_devices()) == 0:
                    raise Exception("No RealSense device detected.")

                config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
                config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
                print("Streaming from RealSense camera.")

            self.pipeline_profile = self.pipeline.start(config)

            # Create and configure a temporal filter
            self.temporal_filter = rs.temporal_filter()
            self.align = rs.align(rs.stream.color)

            # Pass the pipeline and filter to Page3
            self.page3.pipeline = self.pipeline
            self.page3.temporal_filter = self.temporal_filter
            self.page3.align = self.align

            self.goto_page2()

        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
            self.exit_application()

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
    parser.add_argument('--bag', type=str, help='Bag file to stream')
    parser.add_argument('--screen', type=int, default=0, help='Screen to save the calibration file for')
    args = parser.parse_args()

    app = QApplication(sys.argv)

    screens = QApplication.screens()
    if args.display < 0 or args.display >= len(screens):
        print(f"Display index {args.display} is out of range. Using default display 0.")
        args.display = 0
    screen = screens[args.display]

    window = MainWindow(args)

    window.setGeometry(screen.availableGeometry())
    window.showFullScreen()
    window.windowHandle().setScreen(screen)
    sys.exit(app.exec())
