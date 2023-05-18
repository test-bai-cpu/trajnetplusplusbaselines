import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from .myutils import read_human_traj_data, pol2cart

from pprint import pprint


def plot_human_traj_motion(human_traj_file, person_id):
    human_traj_data = read_human_traj_data(human_traj_file, person_id)
    (u, v) = pol2cart(human_traj_data[:, 3], human_traj_data[:, 4])
    color = human_traj_data[:, 0]
    plt.quiver(human_traj_data[:, 1], human_traj_data[:, 2], u, v, color, angles='xy', scale_units='xy', scale=1)
    # plt.show()


def plot_predicted_traj_motion(predicted_traj):
    # predicted_traj = predicted_traj[0:15, :]
    # pprint(predicted_traj)
    # print(predicted_traj[:, 3])
    # print("------")
    (u, v) = pol2cart(predicted_traj[:, 3], predicted_traj[:, 4])
    # color = np.arange(predicted_traj.shape[0])
    # color = predicted_traj[:, 0]
    color = 800
    plt.quiver(predicted_traj[:, 1], predicted_traj[:, 2], u, v, color, angles='xy', scale_units='xy', scale=1)
    # plt.show()


def plot_human_traj(human_traj_data):
    plt.plot(human_traj_data[:, 1] / 0.0247, human_traj_data[:, 2] / 0.0247, color="r")
    plt.scatter(human_traj_data[:, 1] / 0.0247, human_traj_data[:, 2] / 0.0247, marker='o', alpha=1, color="r", s=10)


def plot_human_traj_thor(human_traj_data):
    plt.plot(human_traj_data[8:, 1], human_traj_data[8:, 2], 'r', alpha=1, linewidth=5, label="Ground truth")

def plot_human_traj_v2(human_traj_data):
    plt.plot(human_traj_data[8:, 1], human_traj_data[8:, 2], color="r")
    plt.scatter(human_traj_data[8:, 1], human_traj_data[8:, 2], marker='o', alpha=1, color="r", s=10)
    # plt.scatter(human_traj_data[8:209, 1], human_traj_data[8:209, 2], marker='o', alpha=1, color="r", s=10, label="Ground truth")

def plot_human_traj_edin(human_traj_data):
    plt.plot(human_traj_data[8:, 1] / 0.0247, human_traj_data[8:, 2] / 0.0247, color="r")
    plt.scatter(human_traj_data[8:, 1] / 0.0247, human_traj_data[8:, 2] / 0.0247, marker='o', alpha=1, color="r", s=10)
    # plt.scatter(human_traj_data[8:209, 1], human_traj_data[8:209, 2], marker='o', alpha=1, color="r", s=10, label="Ground truth")

def plot_predicted_traj(predicted_traj):
    color = predicted_traj[:, 0]

    (u, v) = pol2cart(predicted_traj[:, 3], predicted_traj[:, 4])

    # rho = np.ones(len(color)) * 0.2
    # (u, v) = pol2cart(rho, predicted_traj[:, 4])

    plt.quiver(predicted_traj[:, 1], predicted_traj[:, 2], u, v, color, angles='xy', scale_units='xy', scale=1)
    # plt.plot(predicted_traj[:, 1], predicted_traj[:, 2], 'b')


def plot_all_predicted_trajs(total_predicted_motion_list, observed_tracklet_length=8):
    for predicted_traj in total_predicted_motion_list:
        # color = predicted_traj[:, 0]
        time_list = predicted_traj[:, 0]
        # print(list(time_list / time_list.sum()))

        (u, v) = pol2cart(predicted_traj[:, 3], predicted_traj[:, 4])

        # rho = np.ones(len(color)) * 0.2
        # (u, v) = pol2cart(rho, predicted_traj[:, 4])

        # plt.quiver(predicted_traj[:, 1], predicted_traj[:, 2], u, v, color, angles='xy', scale_units='xy', scale=1)
        plt.plot(predicted_traj[:, 1], predicted_traj[:, 2], 'b', alpha=1)
        for i in range(0, observed_tracklet_length):
            plt.scatter(predicted_traj[i, 1], predicted_traj[i, 2], color="limegreen", marker="o", s=10)
        plt.scatter(predicted_traj[observed_tracklet_length:, 1], predicted_traj[observed_tracklet_length:, 2], color="b", marker="o", s=10)

def plot_all_predicted_trajs_edin(total_predicted_motion_list, observed_tracklet_length=8):
    for predicted_traj in total_predicted_motion_list:
        # color = predicted_traj[:, 0]
        time_list = predicted_traj[:, 0]
        # print(list(time_list / time_list.sum()))

        (u, v) = pol2cart(predicted_traj[:, 3], predicted_traj[:, 4])

        # rho = np.ones(len(color)) * 0.2
        # (u, v) = pol2cart(rho, predicted_traj[:, 4])

        # plt.quiver(predicted_traj[:, 1], predicted_traj[:, 2], u, v, color, angles='xy', scale_units='xy', scale=1)
        plt.plot(predicted_traj[:, 1] / 0.0247, predicted_traj[:, 2] / 0.0247, 'b', alpha=1)
        for i in range(0, observed_tracklet_length):
            plt.scatter(predicted_traj[i, 1] / 0.0247, predicted_traj[i, 2] / 0.0247, color="limegreen", marker="o", s=10)
        plt.scatter(predicted_traj[observed_tracklet_length:, 1] / 0.0247, predicted_traj[observed_tracklet_length:, 2] / 0.0247, color="b", marker="o", s=10)


def plot_predicted_traj_edin_trajnet(pred_path):
    ## Plot pred part in blue
    x = np.array([r.x for r in pred_path]) / 0.0247
    y = 460 - np.array([r.y for r in pred_path]) / 0.0247
    plt.plot(x, y, 'b', alpha=1)
    plt.scatter(x, y, color="b", marker="^", s=10)

    
def plot_gt_pred_edin_trajnet(gt_traj, pred_path, obs_length):
    frame_list = [r.frame for r in gt_traj]
    # color = frame_list[:, 0]
    # print(list(frame_list / frame_list.sum()))

    ## Plot pred part in blue
    x = np.array([r.x for r in pred_path]) / 0.0247
    y = 460 - np.array([r.y for r in pred_path]) / 0.0247
    plt.plot(x, y, 'b', alpha=1)
    plt.scatter(x, y, color="b", marker="o", s=10)

    x = np.array([gt_traj[obs_length-1].x, pred_path[0].x]) / 0.0247
    y = 460 - np.array([gt_traj[obs_length-1].y, pred_path[0].y]) / 0.0247
    plt.plot(x, y, 'b', alpha=1)

    ## Plot observed part in green

    x = np.array([r.x for r in gt_traj[:obs_length]]) / 0.0247
    y = 460 - np.array([r.y for r in gt_traj[:obs_length]]) / 0.0247
    plt.plot(x, y, 'g', alpha=1)
    plt.scatter(x, y, color="g", marker="o", s=10)
    
    ## Plot other gt part in red
    
    x = np.array([r.x for r in gt_traj[obs_length-1:]]) / 0.0247
    y = 460 - np.array([r.y for r in gt_traj[obs_length-1:]]) / 0.0247
    plt.plot(x, y, 'r', alpha=1)
    plt.scatter(x, y, color="r", marker="o", s=10)

def plot_gt_pred_atc_trajnet(gt_traj, pred_path, obs_length):
    frame_list = [r.frame for r in gt_traj]
    # color = frame_list[:, 0]
    # print(list(frame_list / frame_list.sum()))

    ## Plot pred part in blue
    x = np.array([r.x for r in pred_path])
    y = np.array([r.y for r in pred_path])
    plt.plot(x, y, 'b', alpha=1)
    plt.scatter(x, y, color="b", marker="o", s=10)

    x = np.array([gt_traj[obs_length-1].x, pred_path[0].x])
    y = np.array([gt_traj[obs_length-1].y, pred_path[0].y])
    plt.plot(x, y, 'b', alpha=1)

    ## Plot observed part in green
    x = np.array([r.x for r in gt_traj[:obs_length]])
    y = np.array([r.y for r in gt_traj[:obs_length]])
    plt.plot(x, y, 'g', alpha=1)
    plt.scatter(x, y, color="g", marker="o", s=10)
    
    ## Plot other gt part in red
    x = np.array([r.x for r in gt_traj[obs_length-1:]])
    y = np.array([r.y for r in gt_traj[obs_length-1:]])
    plt.plot(x, y, 'r', alpha=1)
    plt.scatter(x, y, color="r", marker="o", s=10)

def plot_only_predicted_trajs_with_horizon_edin(total_predicted_motion_list, horizon, observed_tracklet_length=8):
    # only plot and normalize weight for the plotted predicted traj, which are reach the max_time_horizon
    filtered_predicted_trajs = []
    weight_list = []
    for predicted_traj in total_predicted_motion_list:
        if round(predicted_traj[-1,0] - predicted_traj[observed_tracklet_length,0],1) < horizon:
            filtered_predicted_trajs.append(predicted_traj)
            weight_list.append(predicted_traj[-1, -1])
    weight_list = np.array(weight_list)
    weight_list = weight_list / weight_list.sum()
    sorted_weight_list = np.sort(weight_list)[::-1]
    

    colors = ["navy", "mediumblue", "blue", "royalblue", "cornflowerblue", "deepskyblue", "skyblue", "lightskyblue", "powderblue"]
    # There is no part before start position in the predicted trajectory, so don't need to add start length here.
    for j in range(len(filtered_predicted_trajs)):
        predicted_traj = filtered_predicted_trajs[j]
        # color = predicted_traj[:, 0]
        time_list = predicted_traj[:, 0]
        # print(list(time_list / time_list.sum()))

        (u, v) = pol2cart(predicted_traj[:, 3], predicted_traj[:, 4])

        # rho = np.ones(len(color)) * 0.2
        # (u, v) = pol2cart(rho, predicted_traj[:, 4])

        # plt.quiver(predicted_traj[:, 1], predicted_traj[:, 2], u, v, color, angles='xy', scale_units='xy', scale=1)
        # plt.plot(predicted_traj[:, 1] / 0.0247, predicted_traj[:, 2] / 0.0247, 'b', alpha=weight_list[j])
        color_index = np.where(sorted_weight_list==weight_list[j])[0][0]
        if color_index >= len(colors):
            color_index = len(colors) - 1
        plt.plot(predicted_traj[:, 1] / 0.0247, predicted_traj[:, 2] / 0.0247, color=colors[color_index])
        for i in range(0, observed_tracklet_length):
            plt.scatter(predicted_traj[i, 1] / 0.0247, predicted_traj[i, 2] / 0.0247, color="limegreen", marker="o", s=10)
        plt.scatter(predicted_traj[observed_tracklet_length:, 1] / 0.0247, predicted_traj[observed_tracklet_length:, 2] / 0.0247, color="b", marker="o", s=10, alpha=weight_list[j])


def plot_all_predicted_trajs_v2(total_predicted_motion_list, observed_tracklet_length=8):
    color_list = ["b", "darkviolet", "c", "m", "y", "k", "royalblue", "turquoise", "rosybrown", "orange"]
    for traj_index, predicted_traj in enumerate(total_predicted_motion_list):
        # color = predicted_traj[:, 0]
        time_list = predicted_traj[:, 0]
        # print(list(time_list / time_list.sum()))
        shape = predicted_traj.shape
        (u, v) = pol2cart(predicted_traj[:, 3], predicted_traj[:, 4])

        # rho = np.ones(len(color)) * 0.2
        # (u, v) = pol2cart(rho, predicted_traj[:, 4])

        # plt.quiver(predicted_traj[:, 1], predicted_traj[:, 2], u, v, color, angles='xy', scale_units='xy', scale=1)
        for i in range(0, observed_tracklet_length):
            plt.scatter(predicted_traj[i, 1], predicted_traj[i, 2], color="limegreen", marker="o", s=10)
        plt.scatter(predicted_traj[observed_tracklet_length:, 1], predicted_traj[observed_tracklet_length:, 2], color="b", marker="o", s=10)
        # for i in range(observed_tracklet_length, shape[0]):
        #     total = shape[0] - observed_tracklet_length
        #     plt.scatter(predicted_traj[i, 1], predicted_traj[i, 2], color="b", marker="o", alpha=1-(i-observed_tracklet_length)/total, s=50)

def plot_all_predicted_trajs_v3(total_predicted_motion_list, observed_tracklet_length):
    color_list = ["b", "darkviolet", "c", "m", "y", "k", "royalblue", "turquoise", "rosybrown", "orange"]
    for traj_index, predicted_traj in enumerate(total_predicted_motion_list):
        # color = predicted_traj[:, 0]
        time_list = predicted_traj[:, 0]
        # print(list(time_list / time_list.sum()))
        shape = predicted_traj.shape
        (u, v) = pol2cart(predicted_traj[:, 3], predicted_traj[:, 4])

        # rho = np.ones(len(color)) * 0.2
        # (u, v) = pol2cart(rho, predicted_traj[:, 4])

        # plt.quiver(predicted_traj[:, 1], predicted_traj[:, 2], u, v, color, angles='xy', scale_units='xy', scale=1)
        if traj_index == 0:
            plt.scatter(predicted_traj[0, 1], predicted_traj[0, 2], color="limegreen", marker="o", s=50, label="Observations")
        else:
            plt.scatter(predicted_traj[0, 1], predicted_traj[0, 2], color="limegreen", marker="o", s=50)
        for i in range(1, observed_tracklet_length):
            plt.scatter(predicted_traj[i, 1], predicted_traj[i, 2], color="limegreen", marker="o", s=50)
        # plt.scatter(predicted_traj[observed_tracklet_length:, 1], predicted_traj[observed_tracklet_length:, 2], color="b", marker="o", s=50)
        
        if traj_index == 0:
            plt.scatter(predicted_traj[observed_tracklet_length+1, 1], predicted_traj[observed_tracklet_length+1, 2], color="b", marker="o", alpha=1, s=50, label="Predictions")
        else:
            plt.scatter(predicted_traj[observed_tracklet_length+1, 1], predicted_traj[observed_tracklet_length+1, 2], color="b", marker="o", alpha=1, s=50)
        for i in range(observed_tracklet_length+2, shape[0]):
            total = shape[0] - observed_tracklet_length
            plt.scatter(predicted_traj[i, 1], predicted_traj[i, 2], color="b", marker="o", alpha=1-(i-observed_tracklet_length)/(1.5*total), s=50)
            # plt.scatter(predicted_traj[i, 1], predicted_traj[i, 2], color="b", marker="o", alpha=1, s=50)

def plot_all_predicted_trajs_thor(total_predicted_motion_list, observed_tracklet_length):
    color_list = ["b", "darkviolet", "c", "m", "y", "k", "royalblue", "turquoise", "rosybrown", "orange"]
    for traj_index, predicted_traj in enumerate(total_predicted_motion_list):
        # color = predicted_traj[:, 0]
        time_list = predicted_traj[:, 0]
        # print(list(time_list / time_list.sum()))
        shape = predicted_traj.shape
        (u, v) = pol2cart(predicted_traj[:, 3], predicted_traj[:, 4])

        # rho = np.ones(len(color)) * 0.2
        # (u, v) = pol2cart(rho, predicted_traj[:, 4])

        # plt.quiver(predicted_traj[:, 1], predicted_traj[:, 2], u, v, color, angles='xy', scale_units='xy', scale=1)
        for i in range(0, observed_tracklet_length):
            plt.scatter(predicted_traj[i, 1], predicted_traj[i, 2], color="limegreen", marker="o", s=50)
        # plt.scatter(predicted_traj[observed_tracklet_length:, 1], predicted_traj[observed_tracklet_length:, 2], color="b", marker="o", s=50)
        for i in range(observed_tracklet_length, shape[0]):
            total = shape[0] - observed_tracklet_length
            plt.scatter(predicted_traj[i, 1], predicted_traj[i, 2], color="b", marker="o", alpha=1-(i-observed_tracklet_length)/total, s=50)


def plot_atc_quiver(atc_traj):
    color = atc_traj[:, 0]

    (u, v) = pol2cart(atc_traj[:, 3], atc_traj[:, 4])

    # rho = np.ones(len(color)) * 0.2
    # (u, v) = pol2cart(rho, predicted_traj[:, 4])

    plt.quiver(atc_traj[:, 1], atc_traj[:, 2], u, v, color, angles='xy', scale_units='xy', scale=1)
    # plt.plot(predicted_traj[:, 1], predicted_traj[:, 2], 'b')

def plot_cliff_map(cliff_map_data):
    (u, v) = pol2cart(cliff_map_data[:, 3], cliff_map_data[:, 2])
    color = cliff_map_data[:, 2]
    # plt.quiver(cliff_map_data[:, 0], cliff_map_data[:, 1], u, v, color)

    plt.quiver(cliff_map_data[:, 0], cliff_map_data[:, 1], u, v, color, alpha=1, cmap="hsv")
    # plt.show()


def plot_cliff_map_with_weight(cliff_map_data):

    ## Only leave the SWND with largest weight
    max_index_list = []
    
    location = cliff_map_data[0, :2]
    weight = cliff_map_data[0, 8]
    max_weight_index = 0

    for i in range(1, len(cliff_map_data)):
        tmp_location = cliff_map_data[i, :2]
        if (tmp_location == location).all():
            tmp_weight = cliff_map_data[i, 8]
            if tmp_weight > weight:
                max_weight_index = i
                weight = tmp_weight
        else:
            max_index_list.append(max_weight_index)
            location = cliff_map_data[i, :2]
            weight = cliff_map_data[i, 8]
            max_weight_index = i

    max_index_list.append(max_weight_index)

    (u, v) = pol2cart(cliff_map_data[:, 3], cliff_map_data[:, 2])
    weight = cliff_map_data[:, 8]

    colors = cliff_map_data[:, 2]  * 180 / np.pi
    colors = np.append(colors, [0, 360])
    norm = Normalize()
    norm.autoscale(colors)
    colormap = cm.hsv

    for i in range(len(cliff_map_data)):
    # for i in range(200):
        ## For only plot max weight:
        if i in max_index_list:
            plt.quiver(cliff_map_data[i, 0], cliff_map_data[i, 1], u[i], v[i], color=colormap(norm(colors))[i], alpha=1, cmap="hsv",angles='xy', scale_units='xy', scale=0.7)
        ## For only plot one point:
        # if cliff_map_data[i, 0] == 20 and cliff_map_data[i, 1] == -13:
        # plt.quiver(cliff_map_data[i, 0], cliff_map_data[i, 1], u[i], v[i], color=colormap(norm(colors))[i], alpha=weight[i], cmap="hsv",angles='xy', scale_units='xy', scale=1, width=0.004)
        # plt.quiver(cliff_map_data[i, 0], cliff_map_data[i, 1], u[i], v[i], color=colormap(norm(colors))[i], alpha=weight[i], cmap="hsv",angles='xy', scale_units='xy', scale=0.7)


    sm = cm.ScalarMappable(cmap=colormap, norm=norm)
    cbar = plt.colorbar(sm, shrink = 0.5, ticks=[0, 90, 180, 270, 360], fraction=0.05)
    cbar.ax.tick_params(labelsize=10)
    plt.text(99, -18,"Orientation [deg]", rotation='vertical')


def plot_an_arrow(cliff_map_data, axs):

    ## Only leave the SWND with largest weight
    max_index_list = []
    
    location = cliff_map_data[0, :2]
    weight = cliff_map_data[0, 8]
    max_weight_index = 0

    for i in range(1, len(cliff_map_data)):
        tmp_location = cliff_map_data[i, :2]
        if (tmp_location == location).all():
            tmp_weight = cliff_map_data[i, 8]
            if tmp_weight > weight:
                max_weight_index = i
                weight = tmp_weight
        else:
            max_index_list.append(max_weight_index)
            location = cliff_map_data[i, :2]
            weight = cliff_map_data[i, 8]
            max_weight_index = i

    max_index_list.append(max_weight_index)

    (u, v) = pol2cart(cliff_map_data[:, 3], cliff_map_data[:, 2])
    weight = cliff_map_data[:, 8]

    colors = cliff_map_data[:, 2]  * 180 / np.pi
    colors = np.append(colors, [0, 360])
    norm = Normalize()
    norm.autoscale(colors)
    colormap = cm.hsv

    count = 0
    for i in range(len(cliff_map_data)):
        ## For only plot max weight:
        # if i in max_index_list:
        ## For only plot one point:
        if cliff_map_data[i, 0] == 22 and cliff_map_data[i, 1] == -13:
            axs.quiver(0, 0, u[i], v[i], color=colormap(norm(colors))[i], alpha=weight[i], cmap="hsv", scale=2.5, width=0.1, headwidth=2, headlength=2.2, headaxislength=2)
            # axs.text(0,count,"weight = " + str(weight[i]), fontsize=15)
            count += 0.01

    # sm = cm.ScalarMappable(cmap=colormap, norm=norm)
    # cbar = plt.colorbar(sm, shrink = 0.5, ticks=[0, 90, 180, 270, 360], fraction=0.05)
    # cbar.ax.tick_params(labelsize=10)
    # plt.text(99, -18,"Orientation [deg]", rotation='vertical')

    return axs


def plot_cliff_map_edin(cliff_map_data):
    (u, v) = pol2cart(cliff_map_data[:, 3] / 0.0247, cliff_map_data[:, 2])
    color = cliff_map_data[:, 2]

    plt.quiver(cliff_map_data[:, 0] / 0.0247, cliff_map_data[:, 1] / 0.0247, u, v, color, alpha=1, cmap="hsv")


def plot_cliff_map_edin_trajnet(cliff_map_data):
    (u, v) = pol2cart(cliff_map_data[:, 3] / 0.0247, cliff_map_data[:, 2])
    color = cliff_map_data[:, 2]

    plt.quiver(cliff_map_data[:, 0] / 0.0247, cliff_map_data[:, 1] / 0.0247, u, v, color, alpha=1, cmap="hsv")


def plot_cliff_map_v2(cliff_map_data):
    (u, v) = pol2cart(cliff_map_data[:, 3], cliff_map_data[:, 2])
    colors = cliff_map_data[:, 2]  * 180 / np.pi
    colors = np.append(colors, [0, 360])
    # plt.quiver(cliff_map_data[:, 0], cliff_map_data[:, 1], u, v, color)

    norm = Normalize()
    norm.autoscale(colors)
    colormap = cm.hsv
    # plt.quiver(cliff_map_data[:, 0], cliff_map_data[:, 1], u, v, color=colormap(norm(colors)), angles='xy', scale_units='xy', scale=1)
    plt.quiver(cliff_map_data[:, 0], cliff_map_data[:, 1], u, v, colors, alpha=1, cmap="hsv")
    sm = cm.ScalarMappable(cmap=colormap, norm=norm)
    cbar = plt.colorbar(sm, shrink = 0.5, ticks=[0, 90, 180, 270, 360], fraction=0.05)
    cbar.ax.tick_params(labelsize=10)
    plt.text(99, -18,"Orientation [deg]", rotation='vertical')
    # plt.text(16.9, -1.8,"Orientation [deg]", {'fontsize': 15}, rotation='vertical')
    # plt.show()

def plot_cliff_map_v3(cliff_map_data):
    (u, v) = pol2cart(cliff_map_data[:, 3], cliff_map_data[:, 2])
    colors = cliff_map_data[:, 2]  * 180 / np.pi
    colors = np.append(colors, [0, 360])
    # plt.quiver(cliff_map_data[:, 0], cliff_map_data[:, 1], u, v, color)

    norm = Normalize()
    norm.autoscale(colors)
    colormap = cm.hsv
    plt.quiver(cliff_map_data[:, 0], cliff_map_data[:, 1], u, v, color=colormap(norm(colors)), angles='xy', scale_units='xy', scale=1)
    sm = cm.ScalarMappable(cmap=colormap, norm=norm)
    # plt.colorbar(sm, shrink = 0.6, ticks=[0, 90, 180, 270, 360], fraction=0.025)
    # plt.text(-2, 0.5,"Orientation [deg]", rotation='vertical')
    # plt.show()

def plot_SWGMM(cliff_map_data):
    (u, v) = pol2cart(cliff_map_data[:, 3], cliff_map_data[:, 2])
    color = cliff_map_data[:, 2]
    # plt.quiver(cliff_map_data[:, 0], cliff_map_data[:, 1], u, v, color)

    # plt.quiver(cliff_map_data[:, 0], cliff_map_data[:, 1], u, v, color="r", angles='xy', scale_units='xy', scale=1)
    plt.quiver(cliff_map_data[:, 0], cliff_map_data[:, 1], u, v, color="r")
    # plt.show()
