import sys
import pyrealsense2 as rs
from PySide6.QtWidgets import QApplication, QMainWindow, QStackedWidget, QMessageBox
from PySide6.QtGui import QScreen
from page1 import Page1
from page2 import Page2
from page3 import Page3
from page4 import Page4
from page5 import Page5
from calibration_data import CalibrationData
import argparse
from typing import Optional

class MainWindow(QMainWindow):
    def __init__(self, args: argparse.Namespace, screen: QScreen) -> None:
        self.args = args
        if args.bag:
            self.bag_file = args.bag
        else:
            self.bag_file = None

        if not args.dir:
            args.dir = None

        super().__init__()

        self.stacked_widget = QStackedWidget()
        self.setCentralWidget(self.stacked_widget)

        self.pipeline: Optional[rs.pipeline] = None

        self.calibration_data = CalibrationData()

        # Create instances of pages
        self.page2 = Page2(self, self.goto_page3, self.exit_application, args.auto_progress)
        self.page3 = Page3(self, self.goto_page4, self.exit_application, self.pipeline, args.auto_progress, self.calibration_data, screen)
        self.page4 = Page4(self, self.goto_page5, self.exit_application, self.pipeline, args.auto_progress, args.ir_low_exposure, self.calibration_data)
        self.page5 = Page5(self, self.exit_application, args.auto_progress, args.screen, args.dir, screen)

        self.stacked_widget.addWidget(self.page2)
        self.stacked_widget.addWidget(self.page3)
        self.stacked_widget.addWidget(self.page4)
        self.stacked_widget.addWidget(self.page5)

        if self.bag_file is None:
            self.page1 = Page1(self, self.init_pipeline, self.exit_application)
            self.stacked_widget.addWidget(self.page1)
            self.stacked_widget.setCurrentWidget(self.page1)
        else:
            self.init_pipeline()

    def init_pipeline(self) -> None:
        # Initialize the RealSense pipeline
        self.pipeline = rs.pipeline()
        config = rs.config()

        try:
            motion_support = False

            if self.bag_file:
                config.enable_device_from_file(self.bag_file)
                print(f"Streaming from bag file: {self.bag_file}")
                self.pipeline_profile = self.pipeline.start(config)
            else:
                # Check if any RealSense devices are connected
                context = rs.context()
                devices = context.query_devices()
                if len(devices) == 0:
                    raise Exception("No RealSense device detected.")

                for dev in devices:
                    for sensor in dev.query_sensors():
                        if sensor.is_motion_sensor():
                            motion_support = True

                config.enable_stream(rs.stream.color, 1280, 800, rs.format.bgr8, 15)
                config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 15)
                config.enable_stream(rs.stream.infrared, 1, 848, 480, rs.format.y8, 15)

                if motion_support:
                    config.enable_stream(rs.stream.accel)
                    config.enable_stream(rs.stream.gyro)

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
            
            self.spatial_filter = rs.spatial_filter()
            self.spatial_filter.set_option(rs.option.filter_magnitude, 2)
            self.spatial_filter.set_option(rs.option.filter_smooth_alpha, .5)
            self.spatial_filter.set_option(rs.option.filter_smooth_delta, 20)

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
            self.page3.spatial_filter = self.spatial_filter
            self.page3.temporal_filter = self.temporal_filter
            self.page3.hole_filter = self.hole_filter
            self.page3.align = self.align
            self.page3.motion_support = motion_support

            self.page4.pipeline = self.pipeline
            self.page4.pipeline_profile = self.pipeline_profile

            # Pass the pipeline to Page2
            self.page2.pipeline = self.pipeline

            self.goto_page2()

        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
            self.exit_application()

    def goto_page2(self) -> None:
        self.page2.start_steps()
        self.stacked_widget.setCurrentWidget(self.page2)

    def goto_page3(self) -> None:
        self.page3.start()
        self.stacked_widget.setCurrentWidget(self.page3)
    
    def goto_page4(self) -> None:
        self.page4.start()
        self.stacked_widget.setCurrentWidget(self.page4)

    def goto_page5(self) -> None:
        self.page5.start(self.calibration_data)
        self.stacked_widget.setCurrentWidget(self.page5)

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
    parser.add_argument('--ir-high-exposure', default=1500, type=float, help='IR camera exposure to use when capturing the screen')
    parser.add_argument('--ir-low-exposure', default=100, type=float, help='IR camera exposure to use when capturing the markers')
    parser.add_argument('--rgb-exposure', default=1500, type=float, help='RGB camera exposure')
    parser.add_argument('--laser-power', default=150, type=float, help='Laser dot grid projector power (0-360)')
    args = parser.parse_args()

    app = QApplication(sys.argv[:1])

    screens = QApplication.screens()
    if args.display < 0 or args.display >= len(screens):
        print(f"Display index {args.display} is out of range. Using default display 0.")
        args.display = 0
    screen = screens[args.display]

    w = MainWindow(args, screen)

    w.show()
    w.setScreen(screen)
    w.move(w.screen().geometry().topLeft())
    w.showFullScreen()

    sys.exit(app.exec())
