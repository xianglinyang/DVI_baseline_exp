"""
This is the experiment for baseline umap on nn_preserving, boundary_preserving, inv_preserving, inv_accu, inv_conf_diff, and time
"""
import umap
import os
import torch
import argparse
import evaluate
import sys
import numpy as np
import utils
import time
import json
from scipy.special import softmax


def main(args):
    WORKING_DIR = args.working_dir
    result = list()
    content_path = args.content_path
    sys.path.append(content_path)
    from Model.model import resnet18
    net = resnet18()

    epoch_id = args.epoch_id
    device = torch.device(args.device)

    model_location = os.path.join(content_path, "Model", "Epoch_{:d}".format(epoch_id), "subject_model.pth")

    net.load_state_dict(torch.load(model_location, map_location=device))
    repr_model = torch.nn.Sequential(*(list(net.children())[:-1]))
    repr_model.to(device)
    repr_model.eval()
    fc_model = torch.nn.Sequential(*(list(net.children())[-1:]))
    fc_model.to(device)
    fc_model.eval()

    METHOD = args.method
    if METHOD == "umap" or "pca":
        train_data = np.load(os.path.join(WORKING_DIR,"train_data.npy"))
        test_data = np.load(os.path.join(WORKING_DIR, "test_data.npy"))
        border_points = np.load(os.path.join(WORKING_DIR, "border_points.npy"))
        train_embedding = np.load(os.path.join(WORKING_DIR, "train_embedding.npy"))
        test_embedding = np.load(os.path.join(WORKING_DIR, "test_embedding.npy"))
        border_embedding = np.load(os.path.join(WORKING_DIR, "border_embedding.npy"))
        train_recon = np.load(os.path.join(WORKING_DIR, "train_recon.npy"))
        test_recon = np.load(os.path.join(WORKING_DIR, "test_recon.npy"))

        fitting_data = np.concatenate((train_data, test_data), axis=0)
        fitting_embedding = np.concatenate((train_embedding, test_embedding), axis=0)

        result.append(evaluate.evaluate_proj_nn_perseverance_knn(train_data, train_embedding, 15))
        result.append(evaluate.evaluate_proj_nn_perseverance_knn(train_data, train_embedding, 20))
        result.append(evaluate.evaluate_proj_nn_perseverance_knn(train_data, train_embedding, 30))

        result.append(evaluate.evaluate_proj_nn_perseverance_knn(fitting_data, fitting_embedding, 15))
        result.append(evaluate.evaluate_proj_nn_perseverance_knn(fitting_data, fitting_embedding, 20))
        result.append(evaluate.evaluate_proj_nn_perseverance_knn(fitting_data, fitting_embedding, 30))

        ori_pred = softmax(utils.batch_run(fc_model, torch.from_numpy(train_data).to(device), 10), axis=1)
        new_pred = softmax(utils.batch_run(fc_model, torch.from_numpy(train_recon).to(device), 10), axis=1)
        ori_label = ori_pred.argmax(-1).astype(np.int)
        result.append(evaluate.evaluate_inv_accu(ori_pred.argmax(-1), new_pred.argmax(-1)))
        result.append(evaluate.evaluate_inv_conf(ori_label, ori_pred, new_pred))

        ori_pred = softmax(utils.batch_run(fc_model, torch.from_numpy(test_data).to(device), 10), axis=1)
        new_pred = softmax(utils.batch_run(fc_model, torch.from_numpy(test_recon).to(device), 10), axis=1)
        ori_label = ori_pred.argmax(-1).astype(np.int)
        result.append(evaluate.evaluate_inv_accu(ori_pred.argmax(-1), new_pred.argmax(-1)))
        result.append(evaluate.evaluate_inv_conf(ori_label, ori_pred, new_pred))

        result.append(
            evaluate.evaluate_proj_boundary_perseverance_knn(train_data, train_embedding, border_points, border_embedding,
                                                             15))
        result.append(
            evaluate.evaluate_proj_boundary_perseverance_knn(train_data, train_embedding, border_points, border_embedding,
                                                             20))
        result.append(
            evaluate.evaluate_proj_boundary_perseverance_knn(train_data, train_embedding, border_points, border_embedding,
                                                             30))

        result.append(
            evaluate.evaluate_proj_boundary_perseverance_knn(test_data, test_embedding, border_points, border_embedding,
                                                             15))
        result.append(
            evaluate.evaluate_proj_boundary_perseverance_knn(test_data, test_embedding, border_points, border_embedding,
                                                             20))
        result.append(
            evaluate.evaluate_proj_boundary_perseverance_knn(test_data, test_embedding, border_points, border_embedding,
                                                             30))
        with open(os.path.join(WORKING_DIR, "exp_result.json"), "w") as f:
            json.dump(result, f)
    elif METHOD == "tsne":
        train_data = np.load(os.path.join(WORKING_DIR,"train_data.npy"))
        border_points = np.load(os.path.join(WORKING_DIR, "border_points.npy"))
        train_embedding = np.load(os.path.join(WORKING_DIR, "train_embedding.npy"))
        border_embedding = np.load(os.path.join(WORKING_DIR, "border_embedding.npy"))

        result.append(evaluate.evaluate_proj_nn_perseverance_knn(train_data, train_embedding, 15))
        result.append(evaluate.evaluate_proj_nn_perseverance_knn(train_data, train_embedding, 20))
        result.append(evaluate.evaluate_proj_nn_perseverance_knn(train_data, train_embedding, 30))

        result.append(
            evaluate.evaluate_proj_boundary_perseverance_knn(train_data, train_embedding, border_points, border_embedding,
                                                             15))
        result.append(
            evaluate.evaluate_proj_boundary_perseverance_knn(train_data, train_embedding, border_points, border_embedding,
                                                             20))
        result.append(
            evaluate.evaluate_proj_boundary_perseverance_knn(train_data, train_embedding, border_points, border_embedding,
                                                             30))

        with open(os.path.join(WORKING_DIR, "exp_result.json"), "w") as f:
            json.dump(result, f)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # PROGRAM level args
    parser.add_argument("--content_path", type=str)
    parser.add_argument("--epoch_id", type=int, default=200)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--advance_attack", type=int, default=0, choices=[0, 1])
    parser.add_argument("--working_dir", type=str)
    parser.add_argument("--method", type=str, choices=["umap", "tsne", "pca"])
    args = parser.parse_args()
    main(args)

