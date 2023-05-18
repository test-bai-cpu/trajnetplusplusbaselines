import os
from collections import defaultdict, OrderedDict
import matplotlib.pyplot as plt
from tqdm import tqdm

import trajnetplusplustools

from .plot_figures import plot_gt_pred_edin_trajnet, plot_predicted_traj_edin_trajnet, plot_gt_pred_atc_trajnet

obs_length = 8
pred_length = 158

def plot_traj(gt_file, input_file):
    reader_gt = trajnetplusplustools.Reader(gt_file, scene_type='paths')
    scenes_gt = [s for _, s in reader_gt.scenes()]
    scenes_id_gt = [s_id for s_id, _ in reader_gt.scenes()]

    reader_pred = trajnetplusplustools.Reader(input_file, scene_type='paths')
    scenes_pred = [s for _, s in reader_pred.scenes()]

    for i in tqdm(range(len(scenes_gt))):
        ground_truth = scenes_gt[i]
        primary_tracks_all = [t for t in scenes_pred[i][0] if t.scene_id == scenes_id_gt[i]]
        primary_tracks = [t for t in primary_tracks_all if t.prediction_number == 0]
        
        gt_path = ground_truth[0]
        # print(gt_path)
        pred_path = primary_tracks
        # print(pred_path)
        person_id = primary_tracks[0].pedestrian

        
        ################## For save plot of each prediction ###################################
        plt.clf()
        plt.close('all')
        plt.figure(figsize=(10, 6), dpi=200)
        plt.subplot(111, facecolor='grey')
        img = plt.imread("localization_grid.jpg")
        plt.imshow(img, cmap='gray', vmin=0, vmax=255, extent=[-60, 80, -40, 20])

        plot_gt_pred_atc_trajnet(gt_path, pred_path, obs_length)

        # plt.savefig("results/edin/switch-back/plot-v6/general/" + edin_date + "-" + hour_num_str + "-" + str(person_id) + ".png", bbox_inches='tight')
        # plt.savefig("journal/edin-" + edin_date + "-" + hour_num_str + "-" + str(person_id) + ".pdf", bbox_inches='tight')
        plt.savefig("../plot-lstm/vanilla-atc-100/" + "scene-" + str(i) + "-person-" + str(person_id) + ".png", bbox_inches='tight')
        # plt.show()
        # break
        #######################################################################################
        

# pred_path = "../all_scene_vanilla_edin_v4/obs-8-pred-10/DATA_BLOCK/test_pred/lstm_vanilla_None_modes1"
# gt_path = "../all_scene_vanilla_edin_v4/obs-8-pred-10/DATA_BLOCK/test_private"

# pred_path = "../all_scene_social_edin_v4/obs-8-pred-50/DATA_BLOCK/test_pred/lstm_social_None_modes1"
# gt_path = "../all_scene_social_edin_v4/obs-8-pred-50/DATA_BLOCK/test_private"

# prior_path = "../all_scene_vanilla_atc_long_train_new/obs-8-pred-150-sample-1000"
prior_path = "../all_scene_vanilla_atc_workshop/b0/obs-8-pred-150-sample-100"
pred_path = prior_path + "/" + "DATA_BLOCK/test_pred/lstm_vanilla_None_modes1"
gt_path = prior_path + "/" + "DATA_BLOCK/test_private"

pred_datasets = [f for f in os.listdir(pred_path)]
true_datasets = [f for f in os.listdir(gt_path)]

# print(pred_datasets)
# print("------------------------")
# print(true_datasets)

for date in pred_datasets:
    pred_file = pred_path + "/" + date
    gt_file = gt_path + "/" + date
    print("Now in file: ", pred_file)
    plot_traj(gt_file, pred_file)
    break




# python -m trajnetbaselines.lstm.plot_trajs_atc