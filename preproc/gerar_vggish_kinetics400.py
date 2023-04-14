import numpy as np
import glob
from tqdm import tqdm
import os

kinetics_files = glob.glob("/media/valter/experiments/bit/kinetics400/kinetics_features_bit/*.npy")
output_vggish = "/home/valter/datasets/vggish_kinetics400"


for file in tqdm(kinetics_files):
    try:
        #x = np.load(file)
        #frames = x.shape[0]
        frames = 10
        y = np.zeros((frames, 128))
        with open(os.path.join(output_vggish, file.split("/")[-1]), 'wb') as outf:
            np.save(outf, y)
    except:
        print(file)
        y = np.zeros((1, 128))
        with open(os.path.join(output_vggish, file.split("/")[-1]), 'wb') as outf:
            np.save(outf, y)

print(len(kinetics_files))