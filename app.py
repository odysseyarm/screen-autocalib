from enum import Enum
import sys
from PySide6.QtCore import QThreadPool
from PySide6.QtWidgets import QApplication, QMainWindow, QStackedWidget, QMessageBox
from PySide6.QtGui import QScreen, QCloseEvent
import pyorbbecsdk
import pyrealsense2 as rs
from page1 import Page1
from page2 import Page2
from page3 import Page3
from page4 import Page4
from page5 import Page5
from calibration_data import CalibrationData
import argparse
from typing import Optional
import depth_sensor.interface.pipeline
import depth_sensor.orbbec.pipeline
import depth_sensor.realsense.pipeline

import signal

class MainWindow(QMainWindow):
    pipeline: depth_sensor.interface.pipeline.Pipeline
    threadpool: QThreadPool

    def __init__(self, args: argparse.Namespace, screen: QScreen) -> None:
        self.args = args
        if args.bag:
            self.bag_file = args.bag
        else:
            self.bag_file = None

        if not args.dir:
            args.dir = None
        
        if not args.screen_diagonal:
            args.screen_diagonal = None

        super().__init__()

        self.stacked_widget = QStackedWidget()
        self.setCentralWidget(self.stacked_widget)

        self.pipeline = None

        self.calibration_data = CalibrationData()

        self.threadpool = QThreadPool()

        # Create instances of pages
        self.page2 = Page2(self, self.goto_page3, self.exit_application, args.auto_progress) # type: ignore
        self.page3 = Page3(self, self.goto_page4, self.exit_application, self.pipeline, args.auto_progress, screen)
        self.page4 = Page4(self, self.goto_page5, self.exit_application, self.pipeline, args.auto_progress, args.ir_low_exposure)
        self.page5 = Page5(self, self.exit_application, args.auto_progress, args.screen, args.dir, screen, args.screen_diagonal)

        self.stacked_widget.addWidget(self.page2)
        self.stacked_widget.addWidget(self.page3)
        self.stacked_widget.addWidget(self.page4)
        self.stacked_widget.addWidget(self.page5)

        if self.bag_file is None and args.source == DepthCameraSource.realsense:
            self.page1 = Page1(self, self.init_pipeline, self.exit_application)
            self.stacked_widget.addWidget(self.page1)
            self.stacked_widget.setCurrentWidget(self.page1)
        else:
            self.init_pipeline()

    def init_pipeline(self) -> None:
        # Initialize the pipeline
        match self.args.source:
            case DepthCameraSource.orbbec:
                config = pyorbbecsdk.Config()
                ob_pipeline = pyorbbecsdk.Pipeline()

                profile_list = ob_pipeline.get_stream_profile_list(pyorbbecsdk.OBSensorType.COLOR_SENSOR)
                _color_profile = profile_list.get_video_stream_profile(1920, 0, pyorbbecsdk.OBFormat.RGB, 30)

                profile_list = ob_pipeline.get_stream_profile_list(pyorbbecsdk.OBSensorType.IR_SENSOR)
                _ir_profile = profile_list.get_video_stream_profile(1600, 0, pyorbbecsdk.OBFormat.Y8, 30)

                profile_list = ob_pipeline.get_stream_profile_list(pyorbbecsdk.OBSensorType.DEPTH_SENSOR)
                _depth_profile = profile_list.get_video_stream_profile(1600, 0, pyorbbecsdk.OBFormat.Y16, 30)

                config.enable_stream(_color_profile)
                config.enable_stream(_ir_profile)
                config.enable_stream(_depth_profile)

                # config.set_align_mode(pyorbbecsdk.OBAlignMode.SW_MODE)
                config.set_align_mode(pyorbbecsdk.OBAlignMode.DISABLE)

                self.pipeline = depth_sensor.orbbec.pipeline.Pipeline(ob_pipeline, config)
                self.pipeline.start()
            case DepthCameraSource.realsense:
                _pipeline = rs.pipeline()
                config = rs.config()

                motion_support = False

                self.pipeline_profile: Optional[rs.pipeline_profile] = None

                if self.bag_file:
                    config.enable_device_from_file(self.bag_file)
                    print(f"Streaming from bag file: {self.bag_file}")
                    self.pipeline_profile = _pipeline.start(config)
                else:
                    # Check if any RealSense devices are connected
                    context = rs.context()
                    devices = context.query_devices()
                    if len(devices) == 0:
                        raise Exception("No RealSense device detected.")

                    for dev in devices: # type: ignore
                        for sensor in dev.query_sensors(): # type: ignore
                            if sensor.is_motion_sensor(): # type: ignore
                                motion_support = True

                    config.enable_stream(rs.stream.color, 1280, 800, rs.format.bgr8, 15)
                    config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 15)
                    config.enable_stream(rs.stream.infrared, 1, 848, 480, rs.format.y8, 15)

                    if motion_support:
                        config.enable_stream(rs.stream.accel)
                        config.enable_stream(rs.stream.gyro)

                    print("Streaming from RealSense camera.")

                    self.pipeline_profile = _pipeline.start(config)

                    # Set fixed exposure and disable auto exposure for depth stream
                    _depth_sensor = self.pipeline_profile.get_device().first_depth_sensor()
                    if _depth_sensor:
                        _depth_sensor.set_option(rs.option.depth_units, 0.0001)
                        _depth_sensor.set_option(rs.option.enable_auto_exposure, False)
                        _depth_sensor.set_option(rs.option.hdr_enabled, True)
                        _depth_sensor.set_option(rs.option.laser_power, self.args.laser_power)

                    _color_sensor = self.pipeline_profile.get_device().first_color_sensor()
                    if _color_sensor:
                        _color_sensor.set_option(rs.option.enable_auto_exposure, False)
                        _color_sensor.set_option(rs.option.exposure, self.args.rgb_exposure)
                        _color_sensor.set_option(rs.option.sharpness, 100) 

                self.hdr_merge = rs.hdr_merge()
                
                # Create and configure a temporal filter
                self.temporal_filter = rs.temporal_filter()
                self.temporal_filter.set_option(rs.option.filter_smooth_alpha, .05)
                self.temporal_filter.set_option(rs.option.filter_smooth_delta, 20)
                self.temporal_filter.set_option(rs.option.holes_fill, 7)

                self.spatial_filter = rs.spatial_filter()
                self.spatial_filter.set_option(rs.option.filter_magnitude, 2)
                self.spatial_filter.set_option(rs.option.filter_smooth_alpha, .5)
                self.spatial_filter.set_option(rs.option.filter_smooth_delta, 20)

                self.align = rs.align(rs.stream.depth)

                self.pipeline = depth_sensor.realsense.pipeline.Pipeline(_pipeline)
                self.pipeline._running = True # type: ignore
            case _:
                raise ValueError("Invalid depth camera source string")

        try:
            self.page2.pipeline = self.pipeline
            self.page3.pipeline = self.pipeline
            self.page4.pipeline = self.pipeline

            self.goto_page2()

        except Exception as e:
            QMessageBox.critical(self, "Error", str(e)) # type: ignore
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
        # stopping the pipeline causes the app to hang
        # if hasattr(self, 'pipeline') and self.pipeline and self.pipeline is not None:
        #     try:
        #         self.pipeline.stop()
        #     except Exception as e:
        #         print(f"Error stopping pipeline: {e}")
        #     self.pipeline = None
        # print("stopped pipeline")

        QApplication.closeAllWindows()
        instance = QApplication.instance()
        if instance is not None:
            instance.quit()

    def closeEvent(self, event: QCloseEvent) -> None:
        # stopping the pipeline causes the app to hang
        # if hasattr(self, 'pipeline') and self.pipeline and self.pipeline is not None:
        #     try:
        #         self.pipeline.stop()
        #     except Exception as e:
        #         print(f"Error stopping pipeline: {e}")
        #     self.pipeline = None

        self.page2.closeEvent(event)
        self.page3.closeEvent(event)
        self.page4.closeEvent(event)

        event.accept()

class DepthCameraSource(Enum):
    orbbec = 'orbbec'
    realsense = 'realsense'

    def __str__(self):
        return self.value

def main():
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    parser = argparse.ArgumentParser(description='RealSense Camera GUI')
    parser.add_argument('--display', type=int, default=0, help='Display index to use')
    parser.add_argument('--bag', type=str, help='Bag file to stream')
    parser.add_argument('--screen', type=int, default=0, help='Screen to save the calibration file for')
    parser.add_argument('--dir', type=str, help='Output directory for calibration file')
    parser.add_argument('--auto-progress', default=False, action="store_true", help='Enable auto-progress mode')
    parser.add_argument('--ir-low-exposure', default=100, type=float, help='IR camera exposure to use when capturing the markers')
    parser.add_argument('--rgb-exposure', default=1500, type=float, help='RGB camera exposure')
    parser.add_argument('--laser-power', default=150, type=float, help='Laser dot grid projector power (0-360)')
    parser.add_argument('--screen-diagonal', type=float, help='Screen diagonal size in inches')
    parser.add_argument('--source', default=DepthCameraSource.realsense, type=DepthCameraSource, choices=list(DepthCameraSource))
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

if __name__ == "__main__":
    main()
