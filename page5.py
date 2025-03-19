import numpy as np
from typing import Callable, Optional
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap, QScreen
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton
import cv2
import quaternion
import matplotlib.pyplot as plt
from pathlib import Path
from platformdirs import user_data_dir
from calibration_data import CalibrationData
import io
import json
import time

class Page5(QWidget):
    def __init__(self, parent: QWidget, exit_application: Callable[[], None], auto_progress: bool, screen_id: int, output_dir: Optional[str], screen: QScreen, screen_diagonal: Optional[float]) -> None:
        super().__init__(parent)
        self.screen_size = screen.size()
        self.exit_application = exit_application
        self.results_layout = QVBoxLayout()
        self.done_countdown_time = 10
        self.done_timer: Optional[QTimer] = None
        self.screen_id = screen_id
        self.output_dir = output_dir
        self.auto_progress = auto_progress
        self.screen_diagonal = screen_diagonal
        self.init_ui()

    def init_ui(self) -> None:
        self.setLayout(self.results_layout)

        self.instructions_label = QLabel("Review results and press Done to save and exit.")
        self.instructions_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.results_layout.addWidget(self.instructions_label)

        self.plane_error_label = QLabel("Plane fit error: ?")
        self.plane_error_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.results_layout.addWidget(self.plane_error_label)

        self.matplotlib_label = QLabel()
        self.matplotlib_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.results_layout.addWidget(self.matplotlib_label)

        self.rgb_label = QLabel()
        self.rgb_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.results_layout.addWidget(self.rgb_label)

        self.done_button = QPushButton("Done")
        self.done_button.clicked.connect(self.save_and_exit)
        self.results_layout.addWidget(self.done_button)

        self.exit_button = QPushButton("Exit")
        self.exit_button.clicked.connect(self.exit_application)
        self.results_layout.addWidget(self.exit_button)

        if self.auto_progress:
            self.done_button.setVisible(False)
            self.exit_button.setVisible(False)

    def start_done_countdown(self, success: bool) -> None:
        self.done_timer = QTimer(self)
        self.done_timer.timeout.connect(lambda: self.update_done_countdown(success))
        self.done_timer.start(1000)

    def update_done_countdown(self, success: bool) -> None:
        """Update the countdown based on whether the calibration was successful or not."""
        self.done_countdown_time -= 1
        if success:
            self.instructions_label.setText(f"Done in {self.done_countdown_time}")
        else:
            self.instructions_label.setText(f"Autocalibration unsuccessful. Exiting in {self.done_countdown_time}")

        if self.done_countdown_time <= 0:
            if success:
                self.save_and_exit()
            else:
                self.done_timer.stop()
                self.exit_application()

    def transform_calibration_data(self, calibration_data: CalibrationData) -> CalibrationData:
        diag_mm = self.screen_diagonal * 25.4 # inch to mm
        width_px = self.screen_size.width()
        height_px = self.screen_size.height()
        aspect = width_px / height_px

        desired_height_mm = diag_mm / np.sqrt(aspect**2 + 1)
        desired_width_mm = aspect * desired_height_mm
        print(f"Desired physical dimensions (mm): width = {desired_width_mm:.2f}, height = {desired_height_mm:.2f}")
    
        # we like meters
        desired_width = desired_width_mm / 1000.0
        desired_height = desired_height_mm / 1000.0

        corners = np.array([[0, 0],
                            [1, 0],
                            [1, 1],
                            [0, 1]], dtype=np.float32)
        ones = np.ones((4, 1), dtype=np.float32)
        corners_h = np.hstack([corners, ones])
        mapped = (calibration_data.h_aligned @ corners_h.T).T
        mapped = mapped[:, :2] / mapped[:, 2:3]

        # assumes perfect screen rectangle
        desired_corners = np.array([[0, 0],
                                    [desired_width, 0],
                                    [desired_width, desired_height],
                                    [0, desired_height]], dtype=np.float32)

        H_corr = cv2.getPerspectiveTransform(mapped.astype(np.float32), desired_corners)

        calibration_data.h_aligned = H_corr @ calibration_data.h_aligned

        pts = calibration_data.detected_marker_pattern_aligned_transformed.copy()  # shape (N, 3)
        pts_h = np.hstack([pts[:, :2], np.ones((pts.shape[0], 1), dtype=np.float32)])
        pts_transformed = (H_corr @ pts_h.T).T
        pts_transformed = pts_transformed[:, :2] / pts_transformed[:, 2:3]
        calibration_data.detected_marker_pattern_aligned_transformed[:, :2] = pts_transformed

        H_corr_4x4 = np.eye(4)
        H_corr_4x4[0:3, 0:3] = H_corr
        calibration_data.xy_transformation_matrix_aligned = H_corr_4x4 @ calibration_data.xy_transformation_matrix_aligned

        return calibration_data

    def start(self, calibration_data: CalibrationData) -> None:
        if self.screen_diagonal is not None:
            print("Transforming calibration data to respect screen diagonal and rectangle...")
            calibration_data = self.transform_calibration_data(calibration_data)
        print("Drawing plots...")
        self.draw_plots(calibration_data)
        if self.auto_progress:
            self.start_done_countdown(True)

    def draw_plots(self, calibration_data: CalibrationData) -> None:
        print("Drawing best quad and detected marker pattern...")
        # Draw the best quad on the image
        cv2.polylines(calibration_data.color_image, [calibration_data.best_quad_2d], isClosed=True, color=(0, 255, 0), thickness=2)

        # Draw the detected marker pattern on the image as purple circles
        for i,point in enumerate(calibration_data.detected_marker_pattern_2d):
            cv2.circle(calibration_data.color_image, tuple(point.astype(int)), 5, (255, 0, 255), -1)
            cv2.putText(calibration_data.color_image, str(i), tuple(point.astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        self.plane_error_label.setText(
            f"Plane fit RMSE: {calibration_data.plane_rmse*1000} mm\n"
            f"Plane fit max error: {calibration_data.plane_max_error*1000} mm"
        )

        print("Showing RGB image with detected parts and 3D plot...")
        # Show RGB image with detected parts
        height, width = calibration_data.color_image.shape[:2]
        bytes_per_line = width * 3
        qt_image = QImage(calibration_data.color_image.data, width, height, bytes_per_line, QImage.Format.Format_BGR888)
        pixmap = QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaled(self.screen_size / 3, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.rgb_label.setPixmap(scaled_pixmap)

        print("Showing 3D plot...")
        # Show Matplotlib 3D plot of detected best quad corners
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        for point in calibration_data.best_quad_aligned:
            ax.scatter(point[0], point[2], point[1], c='r', marker='o')  # Swap Y and Z

        # Draw best quad to the plot
        for i in range(4):
            ax.plot([calibration_data.best_quad_aligned[i][0], calibration_data.best_quad_aligned[(i + 1) % 4][0]],
                    [calibration_data.best_quad_aligned[i][2], calibration_data.best_quad_aligned[(i + 1) % 4][2]],
                    [calibration_data.best_quad_aligned[i][1], calibration_data.best_quad_aligned[(i + 1) % 4][1]], 'g')  # Swap Y and Z

        # Now also draw the detected chessboard corner 3d points in the plot
        # for point in calibration_data.points_3d_aligned:
        #     ax.scatter(point[0], point[2], point[1], c='b', marker='x', s=0.5)  # Swap Y and Z
        # display every 1000th point
        for i, point in enumerate(calibration_data.points_3d_aligned):
            if i % 100 == 0:
                ax.scatter(point[0], point[2], point[1], c='b', marker='x', s=0.5)

        for point in calibration_data.detected_marker_pattern_aligned:
            ax.scatter(point[0], point[2], point[1], c='violet', marker='o', s=10)  # Swap Y and Z

        ax.set_xlabel('X')
        ax.set_ylabel('Z')
        ax.set_zlabel('Y')

        ax.invert_zaxis() # Actually inverting y axis
        ax.set_aspect('equal')

        print("Drawing 3D plot to image...")
        fig.canvas.draw()
        width, height = fig.canvas.get_width_height()
        matplotlib_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(height, width, 3)
        qt_image = QImage(matplotlib_image.data, width, height, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaled(self.screen_size / 3, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.matplotlib_label.setPixmap(scaled_pixmap)

        plt.close(fig)

        # todo this shouldn't be in draw_plots
        # set members for saving
        self.homography = np.linalg.inv(calibration_data.h_aligned)
        self.object_points = calibration_data.detected_marker_pattern_aligned_transformed
        self.rotation = np.roll(quaternion.as_float_array(quaternion.from_rotation_matrix(
            calibration_data.xy_transformation_matrix_aligned[:3, :3]
        )), -1)

    def save_and_exit(self) -> None:
        if self.done_timer is not None:
            self.done_timer.stop()

        if self.output_dir:
            _user_data_dir = Path(self.output_dir)
        else:
            _user_data_dir = Path(user_data_dir("odyssey", "odysseyarm", roaming=True))

        screens_dir = _user_data_dir.joinpath("screens")
        if not screens_dir.exists():
            screens_dir.mkdir(parents=True)

        screen_path = screens_dir.joinpath(f"screen_{self.screen_id}.json")

        print(f"Saving calibration data to {screen_path}...")

        with io.open(screen_path, 'w', encoding='utf-8') as f:
            f.write(json.dumps({
                "rotation": self.rotation.tolist(),
                "homography": self.homography.transpose().flatten().tolist(),
                "object_points": self.object_points.tolist(),
            }))
        
        print("Saved. Exiting application.")

        self.exit_application()
