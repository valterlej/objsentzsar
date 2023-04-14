"""
Embeddeder for videos
"""
import warnings
warnings.filterwarnings("ignore")
import os
import numpy as np
import glob
import torch
import pickle
import sys
import bit_pytorch.models as models
from utils import TransformerEmbedder
from dataset import load_object_classes_and_descriptions
from tqdm import tqdm
from sklearn.decomposition import PCA
import json
import random

device = 0
model_name = "BiT-M-R152x2"
model_dir = "data/models/"
#dataset_dir = "/home/valter/datasets/activitynetcaptions_features_bit"
#dataset_dir = "/home/valter/datasets/bit/hmdb51/hmdb51_features_bit_152"
dataset_dir = "/home/valter/Vídeos/hmdb51_features_bit"
#dataset_dir = "/media/valter/experiments/bit/kinetics400/kinetics_features_bit/"
#dataset_dir = "/home/valter/datasets/ucf101_bit_features"
#output_dataset_dir = "/home/valter/datasets/activitynetcaptions_features_bit_obj_embedded"
#output_dataset_dir = "/home/valter/datasets/activitynetcaptions_features_bit_pca500"
#output_dataset_dir = "/home/valter/datasets/bit/hmdb51/hmdb51_features_bit_pca500"
output_dataset_dir = "/home/valter/Vídeos/hmdb51_features_bit_pca500/"
#output_dataset_dir = "/media/valter/experiments/bit/kinetics400/kinetics_features_bit_pca500/"
#output_dataset_dir = "/home/valter/datasets/ucf101_features_bit_pca500"
imagenet_file = "data/models/imagenet21k_wordnet_lemmas.txt"
embedder_name = "paraphrase-distilroberta-base-v2"
embedder = TransformerEmbedder(model_name=embedder_name)

#training_dataset = "data/"
#train_ids = list(json.load(open("data/train.json","r")).keys())


print("Loading imagenet names and descriptions...")
obj_names, obj_descriptions = load_object_classes_and_descriptions(imagenet_file)

#print("Embedding imagenet descriptions")
#obj_embeddings = []
#for d in tqdm(obj_descriptions):
#    obj_embeddings.append(embedder.emb_sentence(d, normalize=False))


torch.backends.cudnn.benchmark = True
device = torch.device("cuda:%d" % device)
torch.set_grad_enabled(False)

print("Loading model...")
model = models.KNOWN_MODELS[model_name]()
model.load_from(np.load(os.path.join(model_dir, f"{model_name}.npz")))
model = model.to(device)
model.eval()



print("Load embeddings for training the PCA...")
files = glob.glob(dataset_dir+"/*.npy")

"""
files_random = glob.glob(dataset_dir+"/*.npy")
random.shuffle(files_random)
list_vid_stacks = []
loaded_files = []
for file in tqdm(files_random):
    fid = file.split("/")[-1][:-4]
    if fid in train_ids:
        try:    
            x = np.load(file)
            list_vid_stacks.append(x)
            loaded_files.append(file)
        except Exception as e:
            print(e)
            print(file)
        if len(list_vid_stacks) >= 7000:
            break
"""

"""
print("Concatenating...")
x = np.concatenate(list_vid_stacks)
del list_vid_stacks
list_vid_stacks = x
print(list_vid_stacks.shape)
print("Learning PCA")
pca = PCA(n_components=500)
pca.fit(list_vid_stacks)
pickle.dump(pca, open("data/bit_pca.pkl","wb"))
"""

pca = pickle.load(open("data/bit_pca.pkl","rb"))



print("Load all embedding to apply the PCA...")
for file in tqdm(files):
    try:    
        x = np.load(file)
        x = pca.transform(x)       
        with open(os.path.join(output_dataset_dir, file.split("/")[-1]), 'wb') as outf:
            np.save(outf, x)    
    except Exception as e:
        print(e)
        print(file)

print("Finish...")