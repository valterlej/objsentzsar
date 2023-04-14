import glob
from tqdm import tqdm

files = glob.glob("/media/valter/experiments/bit/kinetics400/kinetics_features_bit/*.npy")
output_file = open("data/kinetics400/dataset/kinetics400_mdvc_format.csv", "w")
output_file.write(f"video_id\tcaption_pred\tstart\tend\tduration\tcategory_32\tsubs\tphase\tidx\n")
for i, f in enumerate(tqdm(files)):
    file = f.split("/")[-1][:-4]
    line = f"{file}\tPLACEHOLDER\t0\t10\t10\t0\tPLACEHOLDER\tval_1\t{i}\n"
    output_file.write(line)
output_file.close()    
