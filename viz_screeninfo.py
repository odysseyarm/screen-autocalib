#!/usr/bin/env python3
import sys, argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("screen_info_path")
    args = parser.parse_args()

    import json
    import numpy as np
    import matplotlib.pyplot as pp

    with open(args.screen_info_path) as f:
        screen_info = json.load(f)

    h = np.array(screen_info["homography"]).reshape(3, 3)
    h_inv = np.linalg.inv(h)

    fig = pp.figure()
    ax = fig.add_subplot(projection="3d")

    # draw object points
    p = screen_info["object_points"]
    ax.scatter(
        [x[0] for x in p],
        [x[2] for x in p],
        [x[1] for x in p],
    )

    # draw screen bounds
    for i in np.linspace(0, 1, num=2):
        # vertical line
        plot_normalized([[i, 0, 1], [i, 1, 1]], h_inv, ax)
        # horizontal line
        plot_normalized([[0, i, 1], [1, i, 1]], h_inv, ax)

    draw_board(h_inv, ax)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Z Label')
    ax.set_zlabel('Y Label')
    ax.invert_zaxis()
    ax.set_aspect("equal")

    pp.show()


def draw_board(h_inv, ax):
    """Draw reprojected charuco board as a grid."""
    import numpy as np

    board_aspect_ratio = 11 / 7
    # screen_aspect_ratio = calculate_screen_aspect_ratio(h_inv)
    # print(f"{screen_aspect_ratio = }")
    screen_aspect_ratio = 16 / 9
    x_scale = y_scale = 1

    if screen_aspect_ratio > board_aspect_ratio:
        x_scale = board_aspect_ratio / screen_aspect_ratio
    else:
        y_scale = screen_aspect_ratio / board_aspect_ratio

    x_lo = -x_scale / 2 + 0.5
    x_hi = x_scale / 2 + 0.5
    y_lo = -y_scale / 2 + 0.5
    y_hi = y_scale / 2 + 0.5
    for x in np.linspace(x_lo, x_hi, num=12):
        plot_normalized([[x, y_lo, 1], [x, y_hi, 1]], h_inv, ax)
    for y in np.linspace(y_lo, y_hi, num=8):
        plot_normalized([[x_lo, y, 1], [x_hi, y, 1]], h_inv, ax)


def calculate_screen_aspect_ratio(h_inv):
    import numpy as np

    a = (h_inv @ np.transpose([0, 0, 1])).transpose()
    b = (h_inv @ np.transpose([1, 0, 1])).transpose()
    c = (h_inv @ np.transpose([0, 1, 1])).transpose()
    w = np.linalg.norm(b - a)
    h = np.linalg.norm(c - a)
    return w / h


def plot_normalized(ep, h_inv, ax):
    """Plot normalized screen space (unit square) coordinates in plane space."""
    import numpy as np

    ep = (h_inv @ np.transpose(ep)).transpose()
    ep /= ep[:, 2].reshape(-1, 1)
    ax.plot(ep[:, 0], [0, 0], ep[:, 1], c=(0, 0.5, 0, 0.5))


if __name__ == "__main__":
    main()
