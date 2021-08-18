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
    # prepare hyperparameters
    OUTPUT_PATH = os.path.join(".", "results")
    if not os.path.exists(OUTPUT_PATH):
        os.mkdir(OUTPUT_PATH)

    CONTENT_PATH = args.content_path
    sys.path.append(CONTENT_PATH)
    if args.dim == 2048:
        from Model.model import resnet50
        net = resnet50()
        dim = 2048
    else:
        from Model.model import resnet18
        net = resnet18()
        dim = 512


    EPOCH = args.epoch_id
    DEVICE = torch.device(args.device)
    METHOD = args.method
    ADVANCE_ATK = args.advance_attack
    OUTPUT_DIR = args.output_dir

    train_path = os.path.join(CONTENT_PATH, "Training_data")
    train_data = torch.load(os.path.join(train_path, "training_dataset_data.pth"), map_location=DEVICE)
    test_path = os.path.join(CONTENT_PATH, "Testing_data")
    test_data = torch.load(os.path.join(test_path, "testing_dataset_data.pth"), map_location=DEVICE)

    if ADVANCE_ATK == 0:
        border_points = os.path.join(CONTENT_PATH, "Model", "Epoch_{:d}".format(EPOCH), "border_centers.npy")
        border_points = np.load(border_points)
    else:
        border_points = os.path.join(CONTENT_PATH, "Model", "Epoch_{:d}".format(EPOCH), "advance_border_centers.npy")
        border_points = np.load(border_points)
    model_location = os.path.join(CONTENT_PATH, "Model", "Epoch_{:d}".format(EPOCH), "subject_model.pth")

    net.load_state_dict(torch.load(model_location, map_location=DEVICE))
    repr_model = torch.nn.Sequential(*(list(net.children())[:-1]))
    repr_model.to(DEVICE)
    repr_model.eval()
    fc_model = torch.nn.Sequential(*(list(net.children())[-1:]))
    fc_model.to(DEVICE)
    fc_model.eval()

    train_data = utils.batch_run(repr_model, train_data, dim)
    test_data = utils.batch_run(repr_model, test_data, dim)
    time_consume = dict()
    if METHOD == "umap":
        OUTPUT_PATH = os.path.join(OUTPUT_PATH,"umap")
        if not os.path.exists(OUTPUT_PATH):
            os.mkdir(OUTPUT_PATH)

        reducer = umap.UMAP(random_state=42)
        t0 = time.time()
        fitting_data = np.concatenate((train_data, border_points), axis=0)
        fitting_embedding = reducer.fit_transform(fitting_data)
        t1 = time.time()
        time_consume["fit_transfrom"]=str(t1-t0)
        train_embedding = fitting_embedding[:len(train_data)]
        border_embedding = fitting_embedding[len(train_data):]

        test_embedding = reducer.transform(test_data)
        t2 = time.time()
        time_consume["transform_test"] = str(t2-t1)

        train_recon = reducer.inverse_transform(train_embedding)
        t3 = time.time()
        time_consume["recon_train"]=str(t3-t2)

        test_recon = reducer.inverse_transform(test_embedding)
        t4 = time.time()
        time_consume["recon_test"]=str(t4-t3)

        out_dir = os.path.join(OUTPUT_PATH, OUTPUT_DIR)
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

        np.save(os.path.join(out_dir,"train_data.npy"), train_data)
        np.save(os.path.join(out_dir,"test_data.npy"), test_data)
        np.save(os.path.join(out_dir,"border_points.npy"), border_points)
        np.save(os.path.join(out_dir,"train_embedding.npy"), train_embedding)
        np.save(os.path.join(out_dir,"test_embedding.npy"), test_embedding)
        np.save(os.path.join(out_dir,"border_embedding.npy"), border_embedding)
        np.save(os.path.join(out_dir,"train_recon.npy"), train_recon)
        np.save(os.path.join(out_dir,"test_recon.npy"), test_recon)
        with open(os.path.join(out_dir,"time.json"),"w") as f:
            json.dump(time_consume, f)

    elif METHOD == "tsne":
        from sklearn.manifold import TSNE
        # from openTSNE import TSNE
        OUTPUT_PATH = os.path.join(OUTPUT_PATH,"tsne")
        if not os.path.exists(OUTPUT_PATH):
            os.mkdir(OUTPUT_PATH)
        reducer = TSNE(
            n_components=2,
            perplexity=30,
            metric="euclidean",
            n_jobs=8,
            random_state=42,
            verbose=True,
        )
        t0 = time.time()
        fitting_data = np.concatenate((train_data, border_points), axis=0)
        fitting_embedding = reducer.fit_transform(fitting_data)
        t1 = time.time()
        print(t1-t0)
        time_consume["fit_transfrom"] = str(t1-t0)

        train_embedding = fitting_embedding[:len(train_data)]
        border_embedding = fitting_embedding[len(train_data):]

        out_dir = os.path.join(OUTPUT_PATH, OUTPUT_DIR)
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

        np.save(os.path.join(out_dir,"train_data.npy"), train_data)
        np.save(os.path.join(out_dir,"test_data.npy"), test_data)
        np.save(os.path.join(out_dir,"border_points.npy"), border_points)
        np.save(os.path.join(out_dir,"train_embedding.npy"), train_embedding)
        np.save(os.path.join(out_dir,"border_embedding.npy"), border_embedding)
        with open(os.path.join(out_dir,"time.json"), "w") as f:
            json.dump(time_consume, f)

    elif METHOD == "pca":
        OUTPUT_PATH = os.path.join(OUTPUT_PATH,"pca")
        if not os.path.exists(OUTPUT_PATH):
            os.mkdir(OUTPUT_PATH)
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=2)
        t0 = time.time()
        fitting_data = np.concatenate((train_data, border_points), axis=0)
        fitting_embedding = reducer.fit_transform(fitting_data)
        t1 = time.time()

        time_consume["fit_transfrom"] = str(t1-t0)
        train_embedding = fitting_embedding[:len(train_data)]
        border_embedding = fitting_embedding[len(train_data):]

        t2 = time.time()
        test_embedding = reducer.transform(test_data)
        time_consume["transform_test"] = str(t2-t1)

        t3 = time.time()
        train_recon = reducer.inverse_transform(train_embedding)
        t4 = time.time()
        time_consume["recon_train"] = str(t4-t3)
        test_recon = reducer.inverse_transform(test_embedding)
        t5 = time.time()
        time_consume["recon_test"] = str(t5-t4)
        out_dir = os.path.join(OUTPUT_PATH, OUTPUT_DIR)
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

        np.save(os.path.join(out_dir,"train_data.npy"), train_data)
        np.save(os.path.join(out_dir,"test_data.npy"), test_data)
        np.save(os.path.join(out_dir,"border_points.npy"), border_points)
        np.save(os.path.join(out_dir,"train_embedding.npy"), train_embedding)
        np.save(os.path.join(out_dir,"test_embedding.npy"), test_embedding)
        np.save(os.path.join(out_dir,"border_embedding.npy"), border_embedding)
        np.save(os.path.join(out_dir,"train_recon.npy"), train_recon)
        np.save(os.path.join(out_dir,"test_recon.npy"), test_recon)
        with open(os.path.join(out_dir,"time.json"),"w") as f:
            json.dump(time_consume, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # PROGRAM level args
    parser.add_argument("--content_path", type=str)
    parser.add_argument("--epoch_id", type=int, default=200)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--advance_attack", type=int, default=0, choices=[0, 1])
    parser.add_argument("--method", type=str, choices=["umap", "tsne", "pca"])
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--dim", type=int, default=512)
    args = parser.parse_args()
    main(args)

