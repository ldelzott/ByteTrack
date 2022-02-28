# How the ant dataset size influenced the performance of the global system (try increasing the size of the dataset)
#   -> Important : use test datas to test the accuracy of the system

# How the system is working on other videos of ants (How good is the generalization with other ants videos ?)

# Try using the system with vanilla COCO NN to detect sheep

# (See email for modalities about the report schedule)

from tinydb import TinyDB
import numpy as np
import matplotlib.pyplot as plt
import argparse


def make_parser():
    parser = argparse.ArgumentParser("Offline Metrics Visualization")
    parser.add_argument(
        "-json_path",
        "--json_path",
        default=None,
        type=str,
        help="Please input the .json file that contains metric datas",
    )
    parser.add_argument(
        "--dummy_argument",
        dest="dummy_argument",
        default=False,
        action="store_true",
        help="dummy_argument description",
    )
    return parser


def plot_metrics(metrics_1_0s, metrics_1_1s,
                        metrics_2_0s, metrics_2_1s,
                        metrics_5_0s, metrics_5_1s,
                        metrics_6_0s, metrics_6_1s):
    # From https://www.delftstack.com/howto/matplotlib/how-to-plot-in-real-time-using-matplotlib/
    # From https://matplotlib.org/stable/gallery/subplots_axes_and_figures/subplots_demo.html

    figure1, axs = plt.subplots(2, 2, figsize=(12, 8), gridspec_kw={'width_ratios': [1, 1], 'height_ratios': [1, 1]})
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.35, hspace=0.40)
    figure1.suptitle('Offline metrics', size=22)
    figure1.canvas.set_window_title('Offline swarm metrics')
    x_axis_values = np.linspace(0, len(metrics_1_0s), len(metrics_1_0s))

    axs[0, 0].set_title('Centers of mass (x,y)')
    axs[0, 0].set(xlabel='Frames', ylabel='x(Orange),y(green)')
    axs[0, 0].plot(x_axis_values, metrics_1_0s, 'tab:orange')
    axs[0, 0].plot(x_axis_values, metrics_1_1s, 'tab:green')

    # 'self.max_queue_size' define the size of the window for MA
    axs[0, 1].set_title('Moving Average over 7 frames')
    axs[0, 1].set(xlabel='Frames', ylabel='x(Orange),y(green)')
    axs[0, 1].plot(x_axis_values, metrics_2_0s, 'tab:orange')
    axs[0, 1].plot(x_axis_values, metrics_2_1s, 'tab:green')

    axs[1, 0].set_title('Global velocity vector (Dx,Dy)')
    axs[1, 0].set(xlabel='Frames', ylabel='Dx(Blue),Dy(Red)')
    axs[1, 0].plot(x_axis_values, metrics_5_0s, 'tab:blue')
    axs[1, 0].plot(x_axis_values, metrics_5_1s, 'tab:red')

    axs[1, 1].set_title('Moving Average over 7 frames')
    axs[1, 1].set(xlabel='Frames', ylabel='Dx(Blue),Dy(Red)')
    axs[1, 1].plot(x_axis_values, metrics_6_0s, 'tab:blue')
    axs[1, 1].plot(x_axis_values, metrics_6_1s, 'tab:red')

    plt.show()


def prepare_dumped_results(dumped_datas):
    metrics_1_0s = []
    metrics_1_1s = []
    metrics_2_0s = []
    metrics_2_1s = []
    metrics_5_0s = []
    metrics_5_1s = []
    metrics_6_0s = []
    metrics_6_1s = []
    for _, dictionary in enumerate(dumped_datas):
        if dictionary['metric_1_0'] == 'empty':
            metrics_1_0s.append(0)
        else:
            metrics_1_0s.append(dictionary['metric_1_0'])

        if dictionary['metric_1_1'] == "empty":
            metrics_1_1s.append(0)
        else:
            metrics_1_1s.append(dictionary['metric_1_1'])

        if dictionary['metric_2_0'] == "empty":
            metrics_2_0s.append(0)
        else:
            metrics_2_0s.append(dictionary['metric_2_0'])

        if dictionary['metric_2_1'] == "empty":
            metrics_2_1s.append(0)
        else:
            metrics_2_1s.append(dictionary['metric_2_1'])

        if dictionary['metric_5_0'] == "empty":
            metrics_5_0s.append(0)
        else:
            metrics_5_0s.append(dictionary['metric_5_0'])

        if dictionary['metric_5_1'] == "empty":
            metrics_5_1s.append(0)
        else:
            metrics_5_1s.append(dictionary['metric_5_1'])

        if dictionary['metric_6_0'] == "empty":
            metrics_6_0s.append(0)
        else:
            metrics_6_0s.append(dictionary['metric_6_0'])

        if dictionary['metric_6_1'] == "empty":
            metrics_6_1s.append(0)
        else:
            metrics_6_1s.append(dictionary['metric_6_1'])

    plot_metrics(metrics_1_0s, metrics_1_1s,
                        metrics_2_0s, metrics_2_1s,
                        metrics_5_0s, metrics_5_1s,
                        metrics_6_0s, metrics_6_1s)


def open_dumped_results(args):
    if args.json_path is None:
        print("Please provide an absolute path for the data with --json_path [path]")
        return
    else:
        dumped_metrics = TinyDB(args.json_path)
        prepare_dumped_results(dumped_metrics.all())


def main():
    args = make_parser().parse_args()
    open_dumped_results(args)


if __name__ == "__main__":
    main()
