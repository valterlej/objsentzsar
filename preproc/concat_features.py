import glob
import numpy as np
import os
from tqdm import tqdm

#output_dataset_dir = "/home/valter/datasets/activitynetcaptions_features_bit_pca500_obj_embedded"
#output_dataset_dir = "/home/valter/datasets/ucf101_features_bit_pca500_obj_embedded"
#output_dataset_dir = "/home/valter/datasets/bit/hmdb51/hmdb51_features_bit_pca500_obj_embedded/"
output_dataset_dir = "/home/valter/Vídeos/hmdb51_features_bit_pca500_obj_embedded/"
#output_dataset_dir = "/media/valter/experiments/bit/kinetics400/kinetics_features_bit_pca500_obj_embedded/" 

#files_bit = glob.glob("/home/valter/datasets/activitynetcaptions_features_bit_pca500/*.npy")
#files_bit = glob.glob("/home/valter/datasets/ucf101_features_bit_pca500/*.npy")
#files_bit = glob.glob("/home/valter/datasets/bit/hmdb51/hmdb51_features_bit_pca500/*.npy")
files_bit = glob.glob("/home/valter/Vídeos/hmdb51_features_bit_pca500/*.npy")
#files_bit = glob.glob("/media/valter/experiments/bit/kinetics400/kinetics_features_bit_pca500/*.npy")

#files_obj = glob.glob("/home/valter/datasets/activitynetcaptions_features_bit_obj_embedded/*.npy")
#files_obj = glob.glob("/home/valter/datasets/ucf101_features_bit_obj_embedded/*.npy")
#files_obj = glob.glob("/home/valter/datasets/bit/hmdb51/hmdb51_features_bit_obj_embedded/*.npy")
files_obj = glob.glob("/home/valter/Vídeos/hmdb51_features_bit_obj_embedded/*.npy")
#files_obj = glob.glob("/media/valter/experiments/bit/kinetics400/kinetics_features_bit_obj_embedded/*.npy")

files_bit = sorted(files_bit)
files_obj = sorted(files_obj)

for i in tqdm(range(0,len(files_obj))):
    f_bit = files_bit[i].split("/")[-1]
    f_obj = files_obj[i].split("/")[-1]
    if f_bit != f_obj:
        continue

    try:
        feat_bit = np.load(files_bit[i])
        feat_obj = np.load(files_obj[i])
        #print(feat_bit.shape)
        #print(feat_obj.shape)
        feat_new = np.concatenate([feat_bit,feat_obj], axis=1)
        #print(feat_new.shape)

        with open(os.path.join(output_dataset_dir, f_bit), 'wb') as outf:
            np.save(outf, feat_new)   
    except Exception as e:
        print(e)

