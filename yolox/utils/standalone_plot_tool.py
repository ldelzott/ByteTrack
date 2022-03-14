
import numpy as np
import matplotlib.pyplot as plt


def plot_results():
    # From https://www.delftstack.com/howto/matplotlib/how-to-plot-in-real-time-using-matplotlib/
    # From https://matplotlib.org/stable/gallery/subplots_axes_and_figures/subplots_demo.html



    figure1, axs = plt.subplots(2, 5, figsize=(32, 18), gridspec_kw={'width_ratios': [1,1,1,1,1], 'height_ratios': [1, 1]})
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.35, hspace=0.40)
    figure1.suptitle('Training with multiple size datasets', size=22)
    figure1.canvas.set_window_title('Training datas')
    x_axis_values = np.linspace(0, 10, 10)
    txt = "AP_IOU_RANGE(Orange),AP_IOU_50(green),AP_IOU_75(purple),AP_IOU_RANGE_large(red),AR_RANGE(blue)"
    plt.figtext(0.5, 0.01, txt, wrap=True, horizontalalignment='center', fontsize=24)

    S_DATA60_AP_IOU_RANGE = [0,0,0.010,0.044,0.097,0.400,0.504,0.562,0.597,0.611]
    S_DATA60_AP_IOU_50 = [0,0,0.014,0.091,0.193,0.705,0.831,0.875,0.916,0.951]
    S_DATA60_AP_IOU_75 = [0,0,0.012,0.036,0.076,0.412,0.583,0.691,0.710,0.745]
    S_DATA60_AP_IOU_RANGE_large_area = [0,0,0.012,0.069,0.149,0.559,0.659,0.691,0.700,0.693]
    S_DATA60_AR_RANGE = [0,0,0.009,0.066,0.148,0.564,0.661,0.693,0.702,0.696]
    axs[0, 0].set_title('DATASET_60')
    axs[0, 0].set(xlabel='EPOCHS', ylabel='Training Metrics')
    axs[0, 0].plot(x_axis_values, S_DATA60_AP_IOU_RANGE, 'tab:orange')
    axs[0, 0].plot(x_axis_values, S_DATA60_AP_IOU_50, 'tab:green')
    axs[0, 0].plot(x_axis_values, S_DATA60_AP_IOU_75, 'tab:purple')
    axs[0, 0].plot(x_axis_values, S_DATA60_AP_IOU_RANGE_large_area, 'tab:red')
    axs[0, 0].plot(x_axis_values, S_DATA60_AR_RANGE, 'tab:blue')

    S_DATA120_AP_IOU_RANGE = [0,0.091,0.397,0.588,0.630,0.651,0.631,0.658,0.652,0.634]
    S_DATA120_AP_IOU_50 = [0,0.162,0.675,0.931,0.982,0.983,0.984,0.975,0.972,0.972]
    S_DATA120_AP_IOU_75 = [0,0.090,0.458,0.714,0.791,0.798,0.776,0.822,0.807,0.774]
    S_DATA120_AP_IOU_RANGE_large_area = [0,0.111,0.510,0.699,0.702,0.720,0.706,0.726,0.722,0.707]
    S_DATA120_AR_RANGE = [0,0.151,0.541,0.701,0.703,0.722,0.708,0.729,0.723,0.709]
    axs[0, 1].set_title('DATASET_120')
    axs[0, 1].set(xlabel='EPOCHS',
                 ylabel='Training Metrics')
    axs[0, 1].plot(x_axis_values, S_DATA120_AP_IOU_RANGE, 'tab:orange')
    axs[0, 1].plot(x_axis_values, S_DATA120_AP_IOU_50, 'tab:green')
    axs[0, 1].plot(x_axis_values, S_DATA120_AP_IOU_75, 'tab:purple')
    axs[0, 1].plot(x_axis_values, S_DATA120_AP_IOU_RANGE_large_area, 'tab:red')
    axs[0, 1].plot(x_axis_values, S_DATA120_AR_RANGE, 'tab:blue')

    S_DATA180_AP_IOU_RANGE = [0.010,0.413,0.620,0.631,0.615,0.615,0.634,0.647,0.638,0.627]
    S_DATA180_AP_IOU_50 = [0.024,0.717,0.942,0.969,0.980,0.985,0.985,0.976,0.972,0.974]
    S_DATA180_AP_IOU_75 = [0.005,0.446,0.774,0.785,0.741,0.739,0.775,0.817,0.807,0.778]
    S_DATA180_AP_IOU_RANGE_large_area = [0.021,0.569,0.709,0.710,0.694,0.692,0.710,0.714,0.709,0.697]
    S_DATA180_AR_RANGE = [0.024,0.580,0.710,0.713,0.697,0.693,0.711,0.718,0.711,0.698]
    axs[0, 2].set_title('DATASET_180')
    axs[0, 2].set(xlabel='EPOCHS',
                  ylabel='Training Metrics')
    axs[0, 2].plot(x_axis_values, S_DATA180_AP_IOU_RANGE, 'tab:orange')
    axs[0, 2].plot(x_axis_values, S_DATA180_AP_IOU_50, 'tab:green')
    axs[0, 2].plot(x_axis_values, S_DATA180_AP_IOU_75, 'tab:purple')
    axs[0, 2].plot(x_axis_values, S_DATA180_AP_IOU_RANGE_large_area, 'tab:red')
    axs[0, 2].plot(x_axis_values, S_DATA180_AR_RANGE, 'tab:blue')

    S_DATA240_AP_IOU_RANGE = [0.082,0.534,0.636,0.652,0.658,0.618,0.626,0.633,0.638,0.634]
    S_DATA240_AP_IOU_50 = [0.155,0.886,0.963,0.980,0.982,0.981,0.984,0.969,0.968,0.969]
    S_DATA240_AP_IOU_75 = [0.073,0.591,0.801,0.821,0.845,0.751,0.765,0.770,0.774,0.772]
    S_DATA240_AP_IOU_RANGE_large_area = [0.102,0.644,0.717,0.726,0.727,0.697,0.702,0.716,0.721,0.716]
    S_DATA240_AR_RANGE = [0.172,0.664,0.720,0.727,0.727,0.699,0.702,0.718,0.723,0.719]
    axs[0, 3].set_title('DATASET_240')
    axs[0, 3].set(xlabel='EPOCHS',
                  ylabel='Training Metrics')
    axs[0, 3].plot(x_axis_values, S_DATA240_AP_IOU_RANGE, 'tab:orange')
    axs[0, 3].plot(x_axis_values, S_DATA240_AP_IOU_50, 'tab:green')
    axs[0, 3].plot(x_axis_values, S_DATA240_AP_IOU_75, 'tab:purple')
    axs[0, 3].plot(x_axis_values, S_DATA240_AP_IOU_RANGE_large_area, 'tab:red')
    axs[0, 3].plot(x_axis_values, S_DATA240_AR_RANGE, 'tab:blue')

    S_DATA270_AP_IOU_RANGE = [0.162,0.485,0.550,0.613,0.601,0.524,0.582,0.649,0.652,0.653]
    S_DATA270_AP_IOU_50 = [0.315,0.939,0.982,0.980,0.987,0.982,0.988,0.978,0.977,0.978]
    S_DATA270_AP_IOU_75 = [0.156,0.426,0.545,0.709,0.661,0.465,0.618,0.812,0.809,0.820]
    S_DATA270_AP_IOU_RANGE_large_area = [0.295,0.637,0.638,0.686,0.682,0.622,0.667,0.713,0.719,0.718]
    S_DATA270_AR_RANGE = [0.303,0.637,0.639,0.689,0.684,0.623,0.667,0.718,0.721,0.719]
    axs[0, 4].set_title('DATASET_270')
    axs[0, 4].set(xlabel='EPOCHS',
                  ylabel='Training Metrics')
    axs[0, 4].plot(x_axis_values, S_DATA270_AP_IOU_RANGE, 'tab:orange')
    axs[0, 4].plot(x_axis_values, S_DATA270_AP_IOU_50, 'tab:green')
    axs[0, 4].plot(x_axis_values, S_DATA270_AP_IOU_75, 'tab:purple')
    axs[0, 4].plot(x_axis_values, S_DATA270_AP_IOU_RANGE_large_area, 'tab:red')
    axs[0, 4].plot(x_axis_values, S_DATA270_AR_RANGE, 'tab:blue')


    X_DATA60_AP_IOU_RANGE = [0,0,0.006,0.332,0.561,0.654,0.663,0.681,0.681,0.682]
    X_DATA60_AP_IOU_50 = [0,0,0.010,0.523,0.868,0.972,0.980,0.983,0.986,0.987]
    X_DATA60_AP_IOU_75 = [0,0,0.007,0.401,0.671,0.808,0.811,0.865,0.865,0.879]
    X_DATA60_AP_IOU_RANGE_large_area = [0,0,0.006,0.377,0.642,0.724,0.730,0.742,0.739,0.737]
    X_DATA60_AR_RANGE = [0,0,0.004,0.376,0.643,0.726,0.733,0.745,0.743,0.740]
    axs[1, 0].set_title('X_DATASET_60')
    axs[1, 0].set(xlabel='EPOCHS',
                  ylabel='Training Metrics')
    axs[1, 0].plot(x_axis_values, X_DATA60_AP_IOU_RANGE, 'tab:orange')
    axs[1, 0].plot(x_axis_values, X_DATA60_AP_IOU_50, 'tab:green')
    axs[1, 0].plot(x_axis_values, X_DATA60_AP_IOU_75, 'tab:purple')
    axs[1, 0].plot(x_axis_values, X_DATA60_AP_IOU_RANGE_large_area, 'tab:red')
    axs[1, 0].plot(x_axis_values, X_DATA60_AR_RANGE, 'tab:blue')



    X_DATA120_AP_IOU_RANGE = [0,0.288,0.623,0.667,0.637,0.657,0.680,0.699,0.697,0.699]
    X_DATA120_AP_IOU_50 = [0,0.486,0.969,0.982,0.981,0.984,0.985,0.988,0.988,0.988]
    X_DATA120_AP_IOU_75 = [0,0.318,0.755,0.838,0.802,0.836,0.861,0.911,0.901,0.902]
    X_DATA120_AP_IOU_RANGE_large_area = [0,0.335,0.712,0.734,0.718,0.728,0.743,0.755,0.753,0.753]
    X_DATA120_AR_RANGE = [0,0.337,0.714,0.735,0.719,0.729,0.746,0.758,0.756,0.757]
    axs[1, 1].set_title('X_DATASET_120')
    axs[1, 1].set(xlabel='EPOCHS',
                  ylabel='Training Metrics')
    axs[1, 1].plot(x_axis_values, X_DATA120_AP_IOU_RANGE, 'tab:orange')
    axs[1, 1].plot(x_axis_values, X_DATA120_AP_IOU_50, 'tab:green')
    axs[1, 1].plot(x_axis_values, X_DATA120_AP_IOU_75, 'tab:purple')
    axs[1, 1].plot(x_axis_values, X_DATA120_AP_IOU_RANGE_large_area, 'tab:red')
    axs[1, 1].plot(x_axis_values, X_DATA120_AR_RANGE, 'tab:blue')



    X_DATA180_AP_IOU_RANGE = [0.010,0.640,0.672,0.696,0.691,0.685,0.708,0.714,0.708,0.708]
    X_DATA180_AP_IOU_50 = [0.020,0.969,0.989,0.987,0.987,0.987,0.987,0.988,0.988,0.988]
    X_DATA180_AP_IOU_75 = [0.010,0.790,0.887,0.892,0.887,0.889,0.919,0.905,0.908,0.910]
    X_DATA180_AP_IOU_RANGE_large_area = [0.011,0.719,0.733,0.750,0.748,0.743,0.762,0.765,0.759,0.758]
    X_DATA180_AR_RANGE = [0.006,0.722,0.733,0.754,0.752,0.746,0.766,0.768,0.763,0.761]
    axs[1, 2].set_title('X_DATASET_180')
    axs[1, 2].set(xlabel='EPOCHS',
                  ylabel='Training Metrics')
    axs[1, 2].plot(x_axis_values, X_DATA180_AP_IOU_RANGE, 'tab:orange')
    axs[1, 2].plot(x_axis_values, X_DATA180_AP_IOU_50, 'tab:green')
    axs[1, 2].plot(x_axis_values, X_DATA180_AP_IOU_75, 'tab:purple')
    axs[1, 2].plot(x_axis_values, X_DATA180_AP_IOU_RANGE_large_area, 'tab:red')
    axs[1, 2].plot(x_axis_values, X_DATA180_AR_RANGE, 'tab:blue')



    X_DATA240_AP_IOU_RANGE = [0.200,0.653,0.646,0.683,0.675,0.693,0.685,0.720,0.717,0.721]
    X_DATA240_AP_IOU_50 = [0.332,0.985,0.985,0.986,0.985,0.987,0.986,0.988,0.988,0.988]
    X_DATA240_AP_IOU_75 = [0.222,0.803,0.804,0.879,0.971,0.898,0.882,0.936,0.923,0.928]
    X_DATA240_AP_IOU_RANGE_large_area = [0.229,0.720,0.709,0.740,0.736,0.749,0.744,0.773,0.769,0.771]
    X_DATA240_AR_RANGE = [0.233,0.722,0.711,0.742,0.740,0.753,0.747,0.776,0.773,0.776]
    axs[1, 3].set_title('X_DATASET_240')
    axs[1, 3].set(xlabel='EPOCHS',
                  ylabel='Training Metrics')
    axs[1, 3].plot(x_axis_values, X_DATA240_AP_IOU_RANGE, 'tab:orange')
    axs[1, 3].plot(x_axis_values, X_DATA240_AP_IOU_50, 'tab:green')
    axs[1, 3].plot(x_axis_values, X_DATA240_AP_IOU_75, 'tab:purple')
    axs[1, 3].plot(x_axis_values, X_DATA240_AP_IOU_RANGE_large_area, 'tab:red')
    axs[1, 3].plot(x_axis_values, X_DATA240_AR_RANGE, 'tab:blue')



    X_DATA270_AP_IOU_RANGE = [0.338,0.617,0.616,0.630,0.650,0.621,0.633,0.716,0.715,0.717]
    X_DATA270_AP_IOU_50 = [0.560,0.987,0.985,0.986,0.987,0.987,0.988,0.988,0.988,0.989]
    X_DATA270_AP_IOU_75 = [0.379,0.736,0.718,0.780,0.832,0.768,0.796,0.944,0.947,0.939]
    X_DATA270_AP_IOU_RANGE_large_area = [0.391,0.689,0.685,0.696,0.719,0.689,0.699,0.765,0.767,0.766]
    X_DATA270_AR_RANGE = [0.403,0.690,0.688,0.698,0.721,0.691,0.701,0.769,0.771,0.770]
    axs[1, 4].set_title('X_DATASET_270')
    axs[1, 4].set(xlabel='EPOCHS',
                  ylabel='Training Metrics')
    axs[1, 4].plot(x_axis_values, X_DATA270_AP_IOU_RANGE, 'tab:orange')
    axs[1, 4].plot(x_axis_values, X_DATA270_AP_IOU_50, 'tab:green')
    axs[1, 4].plot(x_axis_values, X_DATA270_AP_IOU_75, 'tab:purple')
    axs[1, 4].plot(x_axis_values, X_DATA270_AP_IOU_RANGE_large_area, 'tab:red')
    axs[1, 4].plot(x_axis_values, X_DATA270_AR_RANGE, 'tab:blue')

    plt.show()


def main():
    plot_results()


if __name__ == "__main__":
    main()
