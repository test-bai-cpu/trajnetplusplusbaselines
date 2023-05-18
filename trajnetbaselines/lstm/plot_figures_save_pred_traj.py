import os
from collections import defaultdict, OrderedDict
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import pandas as pd
import trajnetplusplustools

from .plot_figures import plot_gt_pred_edin_trajnet, plot_predicted_traj_edin_trajnet, plot_gt_pred_atc_trajnet

obs_length = 8
pred_length = 158

def export_traj(gt_file, input_file, date):
    reader_gt = trajnetplusplustools.Reader(gt_file, scene_type="paths")
    scenes_gt = [s for _, s in reader_gt.scenes()]
    scenes_id_gt = [s_id for s_id, _ in reader_gt.scenes()]

    reader_pred = trajnetplusplustools.Reader(input_file, scene_type="paths")
    scenes_pred = [s for _, s in reader_pred.scenes()]

    for i in tqdm(range(len(scenes_gt))):
        ground_truth = scenes_gt[i]
        primary_tracks_all = [t for t in scenes_pred[i][0] if t.scene_id == scenes_id_gt[i]]
        primary_tracks = [t for t in primary_tracks_all if t.prediction_number == 0]
        
        gt_traj = ground_truth[0]
        pred_path = primary_tracks
        person_id = primary_tracks[0].pedestrian
        
        print(person_id, type(person_id))


# 9155602
        if person_id == 11365400:
            x = np.array([r.x for r in pred_path])
            y = np.array([r.y for r in pred_path])
            df = pd.DataFrame({"x": x, "y": y})
            df.to_csv("../export-pred-traj-atc/" + date + "-" + str(person_id) + ".csv", index=False)
            
            
        
        




prior_path = "../all_scene_vanilla_atc_long_train_new_pred_150/obs-8-pred-150-sample-1000"
pred_path = prior_path + "/" + "DATA_BLOCK/test_pred/lstm_vanilla_None_modes1"
gt_path = prior_path + "/" + "DATA_BLOCK/test_private"

pred_datasets = [f for f in os.listdir(pred_path)]
true_datasets = [f for f in os.listdir(gt_path)]

print(pred_datasets)
print("------------------------")
print(true_datasets)

date = "1031.ndjson"
pred_file = pred_path + "/" + date
gt_file = gt_path + "/" + date
print("Now in file: ", pred_file)
export_traj(gt_file, pred_file, date.split(".")[0])



# python -m trajnetbaselines.lstm.plot_trajs_atc