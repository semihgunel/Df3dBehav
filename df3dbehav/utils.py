import numpy as np
import glob
import pickle
import os

def read_pose_result_mm(p: str):
    n = np.memmap(get_pose_result_mm_path(p), dtype="float32", mode="r").size // 76
    return np.memmap(
        get_pose_result_mm_path(p), dtype="float32", mode="r", shape=(n, 76)
    )


def get_pose_result_mm_path(p: str) -> str:
    return glob.glob(os.path.join(p, "*pose_result*.mm"))[0]


def read_pose_result(p: str):
    pose_res = pickle.load(open(get_pose_result_path(p), "rb"))
    return np.concatenate(
        [pose_res["points2d"][1][:, :19], pose_res["points2d"][5][:, 19:]], axis=1
    )    


def get_pose_result_path(p: str):
    return glob.glob(os.path.join(p, "*pose_result*.pkl"))[0]
                                                          
                                                          
