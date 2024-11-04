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
        self.args = args
        if args.bag:
            self.bag_file = args.bag
        else:
            self.bag_file = None

        if not args.dir:
            args.dir = None

        if not args.depth_from_markers:
            args.depth_from_markers = False

        super().__init__()

        self.stacked_widget = QStackedWidget()
        self.setCentralWidget(self.stacked_widget)

        self.pipeline: Optional[rs.pipeline] = None

        # Create instances of pages
        self.page2 = Page2(self, self.goto_page3, self.exit_application, args.auto_progress)
        self.page3 = Page3(self, self.exit_application, self.pipeline, args.screen, args.dir, args.auto_progress, args.depth_from_markers, args.ir_low_exposure)

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
                self.pipeline_profile = self.pipeline.start(config)
            else:
                # Check if any RealSense devices are connected
                context = rs.context()
                if len(context.query_devices()) == 0:
                    raise Exception("No RealSense device detected.")

                config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)
                config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
                config.enable_stream(rs.stream.infrared, 1, 848, 480, rs.format.y8, 30)
                # config.enable_stream(rs.stream.accel)
                # config.enable_stream(rs.stream.gyro)
                print("Streaming from RealSense camera.")

                self.pipeline_profile = self.pipeline.start(config)

                # Set fixed exposure and disable auto exposure for depth stream
                depth_sensor = self.pipeline_profile.get_device().first_depth_sensor()
                if depth_sensor:
                    depth_sensor.set_option(rs.option.enable_auto_exposure, 0)
                    depth_sensor.set_option(rs.option.exposure, self.args.ir_high_exposure)
                    depth_sensor.set_option(rs.option.laser_power, self.args.laser_power)

                color_sensor = self.pipeline_profile.get_device().first_color_sensor()
                if color_sensor:
                    color_sensor.set_option(rs.option.enable_auto_exposure, 0)
                    color_sensor.set_option(rs.option.exposure, self.args.rgb_exposure)

            # Create and configure a temporal filter
            self.temporal_filter = rs.temporal_filter()
            self.temporal_filter.set_option(rs.option.filter_smooth_alpha, .05)
            self.temporal_filter.set_option(rs.option.filter_smooth_delta, 20)

            self.hole_filter = rs.hole_filling_filter()
            self.hole_filter.set_option(rs.option.holes_fill, 2)
            self.align = rs.align(rs.stream.depth)

            # Pass the pipeline and filter to Page3
            self.page3.pipeline = self.pipeline
            self.page3.pipeline_profile = self.pipeline_profile
            self.page3.temporal_filter = self.temporal_filter
            self.page3.hole_filter = self.hole_filter
            self.page3.align = self.align

            # Pass the pipeline to Page2
            self.page2.pipeline = self.pipeline

            self.goto_page2()

        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
            self.exit_application()

    def goto_page3(self) -> None:
        self.page3.start()
        self.stacked_widget.setCurrentWidget(self.page3)

    def goto_page2(self) -> None:
        self.page2.start_steps()
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
    parser.add_argument('--dir', type=str, help='Output directory for calibration file')
    parser.add_argument('--auto-progress', default=False, action="store_true", help='Enable auto-progress mode')
    parser.add_argument('--depth-from-markers', default=False, action="store_true", help='Use depth from markers instead of from screen')
    parser.add_argument('--ir-high-exposure', default=1500, type=float, help='IR camera exposure to use when capturing the screen')
    parser.add_argument('--ir-low-exposure', default=100, type=float, help='IR camera exposure to use when capturing the markers')
    parser.add_argument('--rgb-exposure', default=1500, type=float, help='RGB camera exposure')
    parser.add_argument('--laser-power', default=150, type=float, help='Laser dot grid projector power (0-360)')
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
