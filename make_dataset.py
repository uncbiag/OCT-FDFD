import numpy as np
import pandas as pd
import random
from glob import glob
import os.path
from compute_electric_simulation import compute_a_line
from dashboard import *
from electric_funcs import *


def make_annotation_file_layered_medium(directory, wavenumbers, bw, dx, dy):
    """
    Helper function that creates an annotation file from the reflected fields computed
    Arguments:
        directory (string): directory where the ER or reflected fields are
        wavenumbers (array): the wavenumbers for which the reflected field was computed
        bw (float): bandwidth of the source power spectrum
        dx (float): finite difference in the x direction of the grid
        dy (float): finite difference in the y direction of the grid
    """
    files = sorted(glob(os.path.join(directory, "*.npy")))
    files_aline = sorted(glob(os.path.join("Dataset/ALI", '*.npy')))
    columnlist = ["id", "num_layers", "er", "r", "z", "k_0", "k_min", "k_max", "bw", "dx", "dy", "dz", "dk"]
    df = pd.DataFrame(columns=columnlist)
    n = len(files)
    delta_k = wavenumbers[1] - wavenumbers[0]
    num_sample_pts = len(wavenumbers)
    delta_z = np.pi / (2 * delta_k * num_sample_pts)
    for i in range(n):
        ER = np.load(files[i])
        ER1 = ER[21, 20:-20]
        filename = files[i].split("/")[-1]
        id = int(filename[3:-4])
        change_idx = np.where(ER1[:-1] != ER1[1:])[0] + 1
        z = dy * (change_idx)
        er = [ER1[0]]
        r = []
        for j in range(len(change_idx)):
            idx = change_idx[j]
            er_idx = ER1[idx]
            er.append(er_idx)
            n1 = np.sqrt(er[j])
            n2 = np.sqrt(er_idx)
            theta2 = np.arcsin((n1 * np.sin(0)) / n2)
            r_idx = (n1 * np.cos(0) - n2 * np.cos(theta2)) / (n1 * np.cos(0) + n2 * np.cos(theta2))
            r.append(r_idx)
        num_layers = len(z)
        new_row = pd.Series({"id": id, "num_layers": num_layers, "er": er, "r": r, "z": z,
                             "k_0": wavenumbers[int(num_sample_pts/2)-1], "k_min": wavenumbers[0],
                             "k_max": wavenumbers[-1], "bw": bw, "dx": dx, "dy": dy, "dz": delta_z, "dk": delta_k})
        df = pd.concat([df, new_row.to_frame().T], ignore_index=True)
    df.to_csv("reflected_fields_dataset_layers.csv", index=False)


def split_dataset_layered_medium(data_dir, target_dir, annotations_file, outdir="Dataset_Split", split=0.7):
    # Read the data and target files from the directory
    data_files = sorted(glob(os.path.join(data_dir, "*.npy")))
    target_files = sorted(glob(os.path.join(target_dir, "*.npy")))

    # Read annotations file
    df = pd.read_csv(annotations_file)
    columnlist = list(df.columns)

    # Split annotations file
    df_train = pd.DataFrame(columns=columnlist)
    df_test = pd.DataFrame(columns=columnlist)

    # Make test and train directories inside the outdir
    data_train_path = os.path.join(outdir, "train", "data")
    target_train_path = os.path.join(outdir, "train", "target")
    data_test_path = os.path.join(outdir, "test", "data")
    target_test_path = os.path.join(outdir, "test", "target")
    os.makedirs(data_train_path, exist_ok=True)
    os.makedirs(target_train_path, exist_ok=True)
    os.makedirs(data_test_path, exist_ok=True)
    os.makedirs(target_test_path, exist_ok=True)

    # Splitting the dataset into test and train
    for i in range(len(data_files)):
        data_file = np.load(data_files[i])
        target_file = np.load(target_files[i])
        data_file = data_file[:, 21]
        target_file = target_file[21, :]
        p = random.random()
        row = df.iloc[i]
        if p <= split:
            np.save(os.path.join(data_train_path, "{0}.npy".format(int(row["id"]))), data_file)
            np.save(os.path.join(target_train_path, "{0}.npy".format(int(row["id"]))), target_file)
            df_train = pd.concat([df_train, row.to_frame().T], ignore_index=True)
        else:
            np.save(os.path.join(data_test_path, "{0}.npy".format(int(row["id"]))), data_file)
            np.save(os.path.join(target_test_path, "{0}.npy".format(int(row["id"]))), target_file)
            df_test = pd.concat([df_test, row.to_frame().T], ignore_index=True)
    df_train.to_csv(os.path.join(outdir, "train", "annotations_train.csv"), index=False)
    df_test.to_csv(os.path.join(outdir, "test", "annotations_test.csv"), index=False)


def split_dataset_by_numlayers(data_dir, target_dir, annotations_file, numlayers=1, outdir="Dataset_Split", split=0.7):
    # Read the data and target files from the directory
    data_files = sorted(glob(os.path.join(data_dir, "*.npy")))
    target_files = sorted(glob(os.path.join(target_dir, "*.npy")))

    # Read annotations file
    df = pd.read_csv(annotations_file)
    df = df[df.num_layers == numlayers]

    # Split dataset
    df_train = df.sample(frac=split)
    df_test = df.drop(df_train.index)

    # Make test and train directories inside the outdir
    data_train_path = os.path.join(outdir, "train", "data")
    target_train_path = os.path.join(outdir, "train", "target")
    data_test_path = os.path.join(outdir, "test", "data")
    target_test_path = os.path.join(outdir, "test", "target")
    os.makedirs(data_train_path, exist_ok=True)
    os.makedirs(target_train_path, exist_ok=True)
    os.makedirs(data_test_path, exist_ok=True)
    os.makedirs(target_test_path, exist_ok=True)

    for i in range(len(data_files)):
        filename = target_files[i].split("/")[-1]
        file_id = int(filename[3:-4])
        data_file = np.load(data_files[i])
        target_file = np.load(target_files[i])
        data_file = data_file[:, 21]
        target_file = target_file[21, :]
        if file_id in df_train["id"].unique():
            np.save(os.path.join(data_train_path, "{0}.npy".format(file_id)), data_file)
            np.save(os.path.join(target_train_path, "{0}.npy".format(file_id)), target_file)
        elif file_id in df_test["id"].unique():
            np.save(os.path.join(data_test_path, "{0}.npy".format(file_id)), data_file)
            np.save(os.path.join(target_test_path, "{0}.npy".format(file_id)), target_file)
    df_train.to_csv(os.path.join(outdir, "train", "annotations_train.csv"), index=False)
    df_test.to_csv(os.path.join(outdir, "test", "annotations_test.csv"), index=False)


def a_line_dataset(directory, wavenumbers, source_intensity, delta_z):
    files = sorted(glob(os.path.join(directory, "*.npy")))
    n = len(files)
    for i in range(n):
        reflected_field = np.load(files[i])
        a_line = compute_a_line(wavenumbers, source_intensity, reflected_field, delta_z, intensity=True)
        id = int(files[i][-7:-4])
        np.save("Dataset/ALI/AL_{0:03d}.npy".format(id), a_line)


if __name__ == '__main__':
    # Splitting the dataset for MLP
    # split_dataset_layered_medium("Dataset/ALI", "Dataset/ER", "reflected_fields_dataset_layers.csv", "Dataset_Split_ALI")

    # Splitting the dataset with a condition on the number of layers
    split_dataset_by_numlayers("Dataset/ALI", "Dataset/ER", "reflected_fields_dataset_layers.csv", 3, "Dataset_Split_3Layer")

    # Creating the annotations file

    # Computing the necessary parameters to pass to the function
    # wavenumbers, source_intensity = source_spectrum(l0, bw_l)
    # bw_k = (2 * np.pi / (l0 - bw_l / 2)) - (2 * np.pi / (l0 + bw_l / 2))
    # l_min = 2 * np.pi / wavenumbers[-1]
    # l_max = 2 * np.pi / wavenumbers[0]
    # nmax = np.sqrt(ermax)
    # dx = l_min / nmax / NRES
    # dy = l_min / nmax / NRES
    #
    # # Passing arguments to function
    # make_annotation_file_layered_medium("Dataset/ER", wavenumbers, bw_k, dx, dy)