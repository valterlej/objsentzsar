"""
Embeddeder for videos


computa a cada 3 features



"""
import warnings
warnings.filterwarnings("ignore")
import os
import numpy as np
import glob
import torch
import sys
import bit_pytorch.models as models
from text_embedder import TransformerEmbedder
from dataset import load_object_classes_and_descriptions
from tqdm import tqdm

device = 0
model_name = "BiT-M-R152x2"
model_dir = "data/models/"
#dataset_dir = "/home/valter/datasets/activitynetcaptions_features_bit"
dataset_dir = "/home/valter/datasets/bit/hmdb51/hmdb51_features_bit_152"
#output_dataset_dir = "/home/valter/datasets/activitynetcaptions_features_bit_obj_embedded_v2"
#output_dataset_dir = "/home/valter/datasets/activitynetcaptions_features_bit_obj_embedded"
output_dataset_dir = "/home/valter/datasets/bit/hmdb51/hmdb51_features_bit_obj_embedded"
imagenet_file = "data/imagenet21k_wordnet_lemmas.txt"
embedder_name = "paraphrase-distilroberta-base-v2"
embedder = TransformerEmbedder(model_name=embedder_name)


print("Loading imagenet names and descriptions...")
obj_names, obj_descriptions = load_object_classes_and_descriptions(imagenet_file)


print("Embedding imagenet descriptions")
obj_embeddings = []
for d in tqdm(obj_descriptions):
    obj_embeddings.append(embedder.emb_sentence(d, normalize=False))


torch.backends.cudnn.benchmark = True
device = torch.device("cuda:%d" % device)
torch.set_grad_enabled(False)

print("Loading model...")
model = models.KNOWN_MODELS[model_name]()
model.load_from(np.load(os.path.join(model_dir, f"{model_name}.npz")))
model = model.to(device)
model.eval()


print("Embedding videos...")
files = glob.glob(dataset_dir+"/*.npy")

print(len(files))

for file in tqdm(files):
    try:    
        x = np.load(file)
        x = np.expand_dims(x, axis = -1)
        x = np.expand_dims(x, axis = -1)
        x = torch.from_numpy(x).to(device)
        logits = model.head.conv(x)[...,0,0]

        # computar a media dos logits a cada 3 observacoes (a cada 3s)
        new_logits = []
        for i in range(0, logits.size(dim=0), 3):
            logit = torch.mean(logits[i:i+3,:], 0, keepdim=True).data.cpu().numpy()
            new_logits.append(torch.from_numpy(logit))

        logits = torch.cat(new_logits, 0).to(device)       


        x = torch.nn.functional.softmax(logits, dim=1).data.cpu().numpy()
        obj_ids = np.argmax(x, axis=1) # apenas o objeto mais provavel
        
        vid_emb_stack = []
        vid_obj_list = []
        for o in obj_ids:
            vid_emb_stack.append(obj_embeddings[o])
            vid_obj_list.append(obj_names[o])
        
        vid_emb_stack = np.concatenate(vid_emb_stack)
        
        ##print(vid_emb_stack.shape)
        ##print(vid_obj_list)
        #print(x)
        ##print("\n")
        with open(os.path.join(output_dataset_dir, file.split("/")[-1]), 'wb') as outf:
            np.save(outf, vid_emb_stack)        
    except Exception as e :
        print(e)
        print(file)
    
print("Finish...")