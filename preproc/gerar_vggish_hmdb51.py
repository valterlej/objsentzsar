import numpy as np
import glob
from tqdm import tqdm
import os

#hmdb_files = glob.glob("/home/valter/datasets/bit/hmdb51_features_bit_152/*.npy")
hmdb_files = glob.glob("/home/valter/Vídeos/hmdb51_features_bit/*.npy")
output_vggish = "/home/valter/Vídeos/vggish_hmdb51"


for file in tqdm(hmdb_files):
    try:
        x = np.load(file)
        frames = x.shape[0]
        y = np.zeros((frames, 128))
        with open(os.path.join(output_vggish, file.split("/")[-1]), 'wb') as outf:
            np.save(outf, y)
    except:
        print(file)
        y = np.zeros((1, 128))
        with open(os.path.join(output_vggish, file.split("/")[-1]), 'wb') as outf:
            np.save(outf, y)

print(len(hmdb_files))