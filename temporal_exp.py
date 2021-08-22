"""
This is the experiment for baseline umap on temporal_preserving
"""
import json

import umap
import os
import argparse
import numpy as np
import evaluate


def main(args):
    # prepare hyperparameters
    OUTPUT_PATH = os.path.join(".", "results")
    if not os.path.exists(OUTPUT_PATH):
        os.mkdir(OUTPUT_PATH)
    OUTPUT_PATH = os.path.join(OUTPUT_PATH, "temporal")
    if not os.path.exists(OUTPUT_PATH):
        os.mkdir(OUTPUT_PATH)

    CONTENT_PATH = args.content_path
    ADVANCE_ATK = args.advance_attack
    DATASET = args.dataset
    START = args.s
    END = args.e
    PERIOD = args.p

    for epoch in range(START, END + PERIOD, PERIOD):
        if epoch == START:
            prev_embedding = None
        else:
            prev_dir = os.path.join(OUTPUT_PATH, DATASET+"_"+str(epoch-PERIOD))
            prev_embedding = os.path.join(prev_dir, "embedding.npy")
            prev_embedding = np.load(prev_embedding)

        if ADVANCE_ATK == 0:
            border_points = os.path.join(CONTENT_PATH, "Model", "Epoch_{:d}".format(epoch), "border_centers.npy")
            border_points = np.load(border_points)
        else:
            border_points = os.path.join(CONTENT_PATH, "Model", "Epoch_{:d}".format(epoch), "advance_border_centers.npy")
            border_points = np.load(border_points)
        train_data = os.path.join(CONTENT_PATH, "Model", "Epoch_{:d}".format(epoch), "train_data.npy")
        train_data = np.load(train_data)
        test_data = os.path.join(CONTENT_PATH, "Model", "Epoch_{:d}".format(epoch), "test_data.npy")
        test_data = np.load(test_data)
        if prev_embedding is not None:
            reducer = umap.UMAP(random_state=42, init=prev_embedding)
        else:
            reducer = umap.UMAP(random_state=42)
        fitting_data = np.concatenate((train_data, border_points), axis=0)
        fitting_embedding = reducer.fit_transform(fitting_data)
        test_embedding = reducer.transform(test_data)

        out_dir = os.path.join(OUTPUT_PATH, DATASET+"_"+str(epoch))
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        np.save(os.path.join(out_dir, "train_data.npy"), train_data)
        np.save(os.path.join(out_dir, "test_data.npy"), test_data)
        np.save(os.path.join(out_dir, "embedding.npy"), fitting_embedding)
        np.save(os.path.join(out_dir, "test_embedding.npy"), test_embedding)

    t = dict()
    t["temporal_train"] = temporal_preserving_train(args, n_neighbors=15)
    t["temporal_test"] = temporal_preserving_test(args, n_neighbors=15)
    with open("{}.json".format(args.dataset)) as f:
        json.dump(t, f)

def temporal_preserving_train(args, n_neighbors):
    """evalute training temporal preserving property"""
    DATASET = args.dataset
    START = args.s
    END = args.e
    PERIOD = args.p
    OUTPUT_PATH = os.path.join(".", "results", "temporal")

    l = args.train_l
    eval_num = int((END - START) / PERIOD)
    alpha = np.zeros((eval_num, l))
    delta_x = np.zeros((eval_num, l))
    for epoch in range(START+PERIOD, END+1, PERIOD):
        prev_dir = os.path.join(OUTPUT_PATH, DATASET+"_"+str(epoch-PERIOD))
        prev_embedding = np.load(os.path.join(prev_dir, "embedding.npy"))[:l]
        prev_data = np.load(os.path.join(prev_dir, "train_data.npy"))
        curr_dir = os.path.join(OUTPUT_PATH, DATASET+"_"+str(epoch))
        embedding = np.load(os.path.join(curr_dir, "embedding.npy"))[:l]
        curr_data = np.load(os.path.join(curr_dir, "train_data.npy"))

        alpha_ = evaluate.find_neighbor_preserving_rate(prev_data, curr_data, n_neighbors)
        delta_x_ = np.linalg.norm(prev_embedding - embedding, axis=1)

        alpha[int((epoch - START) / PERIOD - 1)] = alpha_
        delta_x[int((epoch - START) / PERIOD - 1)] = delta_x_

    # val_entropy = evaluate_proj_temporal_perseverance_entropy(alpha, delta_x)
    val_corr = evaluate.evaluate_proj_temporal_perseverance_corr(alpha, delta_x)
    return val_corr

def temporal_preserving_test(args, n_neighbors):
    """evalute training temporal preserving property"""
    DATASET = args.dataset
    START = args.s
    END = args.e
    PERIOD = args.p
    OUTPUT_PATH = os.path.join(".", "results", "temporal")

    train_l = args.train_l
    test_l = args.test_l
    l = train_l + test_l
    eval_num = int((END - START) / PERIOD)
    alpha = np.zeros((eval_num, l))
    delta_x = np.zeros((eval_num, l))
    for epoch in range(START+PERIOD, END+1, PERIOD):
        prev_dir = os.path.join(OUTPUT_PATH, DATASET+"_"+str(epoch-PERIOD))
        prev_train_embedding = np.load(os.path.join(prev_dir, "embedding.npy"))[:train_l]
        prev_test = np.load(os.path.join(prev_dir, "test_data.npy"))
        prev_data = np.load(os.path.join(prev_dir, "train_data.npy"))
        prev_test_embedding = np.load(os.path.join(prev_dir, "test_embedding.npy"))
        prev = np.concatenate((prev_data, prev_test), axis=0)
        prev_embedding = np.concatenate((prev_train_embedding, prev_test_embedding), axis=0)

        curr_dir = os.path.join(OUTPUT_PATH, DATASET+"_"+str(epoch))
        curr_train_embedding = np.load(os.path.join(curr_dir, "embedding.npy"))[:train_l]
        curr_data = np.load(os.path.join(curr_dir, "train_data.npy"))
        curr_test = np.load(os.path.join(curr_dir, "test_data.npy"))
        curr_test_embedding = np.load(os.path.join(curr_dir, "test_embedding.npy"))
        curr = np.concatenate((curr_data, curr_test), axis=0)
        curr_embedding = np.concatenate((curr_train_embedding, curr_test_embedding), axis=0)

        alpha_ = evaluate.find_neighbor_preserving_rate(prev, curr, n_neighbors)
        delta_x_ = np.linalg.norm(prev_embedding - curr_embedding, axis=1)

        alpha[int((epoch - START) / PERIOD - 1)] = alpha_
        delta_x[int((epoch - START) / PERIOD - 1)] = delta_x_

    val_corr = evaluate.evaluate_proj_temporal_perseverance_corr(alpha, delta_x)
    return val_corr


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # PROGRAM level args
    parser.add_argument("--content_path", type=str)
    parser.add_argument("--advance_attack", type=int, default=1, choices=[0, 1])
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("-s", type=int, help="starting epoch")
    parser.add_argument("-e", type=int, help="ending epoch")
    parser.add_argument("-p", type=int, help="period")
    parser.add_argument("--train_l", type=int, help="length of training dataset")
    parser.add_argument("--test_l", type=int, help="length of testing dataset")
    args = parser.parse_args()
    main(args)
