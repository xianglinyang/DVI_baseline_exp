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


def main(args):
    result = list()
    content_path = args.content_path
    sys.path.append(content_path)
    from Model.model import resnet18
    net = resnet18()

    epoch_id = args.epoch_id
    device = torch.device(args.device)

    train_path = os.path.join(content_path, "Training_data")
    train_data = torch.load(os.path.join(train_path, "training_dataset_data.pth"), map_location=device)
    test_path = os.path.join(content_path, "Testing_data")
    test_data = torch.load(os.path.join(test_path, "testing_dataset_data.pth"), map_location=device)
    if args.advance_attack == 0:
        border_points = os.path.join(content_path, "Model", "Epoch_{:d}".format(epoch_id), "border_centers.npy")
        border_points = np.load(border_points)
    else:
        border_points = os.path.join(content_path, "Model", "Epoch_{:d}".format(epoch_id), "advance_border_centers.npy")
        border_points = np.load(border_points)
    model_location = os.path.join(content_path, "Model", "Epoch_{:d}".format(epoch_id), "subject_model.pth")

    net.load_state_dict(torch.load(model_location, map_location=device))
    repr_model = torch.nn.Sequential(*(list(net.children())[:-1]))
    repr_model.to(device)
    repr_model.eval()
    fc_model = torch.nn.Sequential(*(list(net.children())[-1:]))
    fc_model.to(device)
    fc_model.eval()

    train_data = utils.batch_run(repr_model, train_data, 512)
    test_data = utils.batch_run(repr_model, test_data, 512)

    reducer = umap.UMAP(random_state=42)
    t0 = time.time()
    fitting_data = np.concatenate((train_data, border_points), axis=0)
    reducer.fit(fitting_data)
    t1 = time.time()

    print(t1-t0)
    train_embedding = reducer.transform(train_data)
    test_embedding = reducer.transform(test_data)
    border_embedding = reducer.transform(border_points)
    t2 = time.time()
    train_recon = reducer.inverse_transform(train_embedding)
    t3 = time.time()
    print(t3-t2)
    test_recon = reducer.inverse_transform(test_embedding)
    t4 = time.time()
    print(t4-t3)

    np.save(os.path.join(content_path, "Model", "Epoch_{:d}".format(epoch_id),"umap_train_data.npy"), train_data)
    np.save(os.path.join(content_path, "Model", "Epoch_{:d}".format(epoch_id),"umap_test_data.npy"), test_data)
    np.save(os.path.join(content_path, "Model", "Epoch_{:d}".format(epoch_id),"umap_border_points.npy"), border_points)
    np.save(os.path.join(content_path, "Model", "Epoch_{:d}".format(epoch_id),"umap_train_embedding.npy"), train_embedding)
    np.save(os.path.join(content_path, "Model", "Epoch_{:d}".format(epoch_id),"umap_test_embedding.npy"), test_embedding)
    np.save(os.path.join(content_path, "Model", "Epoch_{:d}".format(epoch_id),"umap_border_embedding.npy"), border_embedding)
    np.save(os.path.join(content_path, "Model", "Epoch_{:d}".format(epoch_id),"umap_train_recon.npy"), train_recon)
    np.save(os.path.join(content_path, "Model", "Epoch_{:d}".format(epoch_id),"umap_test_recon.npy"), test_recon)


    # result.append(round(t4-t0, 4))
    #
    # fitting_data = np.concatenate((train_data, test_data), axis=0)
    # fitting_embedding = np.concatenate((train_embedding, test_embedding), axis=0)
    #
    # result.append(evaluate.evaluate_proj_nn_perseverance_knn(train_data, train_embedding, 10))
    # result.append(evaluate.evaluate_proj_nn_perseverance_knn(train_data, train_embedding, 15))
    # result.append(evaluate.evaluate_proj_nn_perseverance_knn(train_data, train_embedding, 30))
    #
    # result.append(evaluate.evaluate_proj_nn_perseverance_knn(fitting_data, fitting_embedding, 10))
    # result.append(evaluate.evaluate_proj_nn_perseverance_knn(fitting_data, fitting_embedding, 15))
    # result.append(evaluate.evaluate_proj_nn_perseverance_knn(fitting_data, fitting_embedding, 30))
    #
    # result.append(evaluate.evaluate_inv_nn(train_data, train_recon, n_neighbors=10))
    # result.append(evaluate.evaluate_inv_nn(train_data, train_recon, n_neighbors=15))
    # result.append(evaluate.evaluate_inv_nn(train_data, train_recon, n_neighbors=30))
    #
    # result.append(evaluate.evaluate_inv_nn(fitting_data, fitting_embedding, n_neighbors=10))
    # result.append(evaluate.evaluate_inv_nn(fitting_data, fitting_embedding, n_neighbors=15))
    # result.append(evaluate.evaluate_inv_nn(fitting_data, fitting_embedding, n_neighbors=30))
    #
    # ori_pred = utils.batch_run(fc_model, torch.from_numpy(train_data).to(device), 10)
    # new_pred = utils.batch_run(fc_model, torch.from_numpy(train_recon).to(device), 10)
    # ori_label = ori_pred.argmax(-1).astype(np.int)
    # result.append(evaluate.evaluate_inv_accu(ori_pred.argmax(-1), new_pred.argmax(-1)))
    # result.append(evaluate.evaluate_inv_conf(ori_label, ori_pred, new_pred))
    #
    # ori_pred = utils.batch_run(fc_model, torch.from_numpy(test_data).to(device), 10)
    # new_pred = utils.batch_run(fc_model, torch.from_numpy(test_recon).to(device), 10)
    # ori_label = ori_pred.argmax(-1).astype(np.int)
    # result.append(evaluate.evaluate_inv_accu(ori_pred.argmax(-1), new_pred.argmax(-1)))
    # result.append(evaluate.evaluate_inv_conf(ori_label, ori_pred, new_pred))
    #
    # result.append(
    #     evaluate.evaluate_proj_boundary_perseverance_knn(train_data, train_embedding, border_points, border_embedding,
    #                                                      10))
    # result.append(
    #     evaluate.evaluate_proj_boundary_perseverance_knn(train_data, train_embedding, border_points, border_embedding,
    #                                                      15))
    # result.append(
    #     evaluate.evaluate_proj_boundary_perseverance_knn(train_data, train_embedding, border_points, border_embedding,
    #                                                      30))
    #
    # result.append(
    #     evaluate.evaluate_proj_boundary_perseverance_knn(test_data, test_embedding, border_points, border_embedding,
    #                                                      10))
    # result.append(
    #     evaluate.evaluate_proj_boundary_perseverance_knn(test_data, test_embedding, border_points, border_embedding,
    #                                                      15))
    # result.append(
    #     evaluate.evaluate_proj_boundary_perseverance_knn(test_data, test_embedding, border_points, border_embedding,
    #                                                      30))
    # with open(os.path.join(content_path, "umap_exp_result.json"), "w") as f:
    #     json.dump(result, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # PROGRAM level args
    parser.add_argument("--content_path", type=str)
    parser.add_argument("--epoch_id", type=int, default=200)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--advance_attack", type=int, default=0, choices=[0, 1])
    parser.add_argument("--method", type=str, choices=["umap", "tsne", "pca"])
    args = parser.parse_args()
    main(args)

