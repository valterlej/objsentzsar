import numpy as np
import torch
import os
import glob
import bit_pytorch.models as models
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
import torch.nn as nn
from time import time, strftime, localtime


def scale_as_probabilities(x):
    return (x.T * (1 / np.sum(x, axis=1))).T

def compute_mean_desv(accs, decimal_places=2):
    x = np.asarray(accs)
    return round(np.mean(x),decimal_places), round(np.std(x),decimal_places)


def semantic_embedding(x, embedder):  # replace the equations (4), (5), (6) and (7)  
    return embedder.emb_sentence(x, normalize=False)

def compute_sparsity(x, Tz=10):
    n_objects, _ = x.shape
    ids = np.argsort(-x,axis=1)[:,:Tz]
    mask = np.zeros(x.shape)
    for i in range(n_objects):
        mask[i,ids[i]] = 1
    return x * mask

def process_file_name(file_name):
    special_chars = ["!","(",")","[","]","&",";"]
    replace_chairs = ["\!","\(","\)","\[","\]","\&","\;"]

    for i, sc in enumerate(special_chars):
        file_name = file_name.replace(sc,replace_chairs[i])
    return file_name


from sentence_model import SentenceEmbeddedDataset, SentenceModel
def get_model_pytorch(bert_embedder_size = 768, num_classes=34, hidden_layer_size=32, drop_rate=0.1):
    model = SentenceModel(bert_embedder_size, num_classes, hidden_layer_size, drop_rate)
    return model


def training_deep_supervised_classifier_pytorch(class_ids, samples, n_classes=34, hidden_size=32, epochs=50, batch_size=64, drop_rate=0.1):
    model = get_model_pytorch(samples.shape[1], n_classes, drop_rate=drop_rate, hidden_layer_size=hidden_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001, eps=1e-7, centered=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device, non_blocking=True)
    param_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Sentence model -- param Num: {param_num}')
    
    indices = np.arange(samples.shape[0])
    np.random.shuffle(indices)
    samples = samples[indices]
    class_ids = class_ids[indices]

    pos = int(len(samples) * 0.7)
    train_embeddings = samples[:pos]
    train_class_ids = class_ids[:pos]
    val_embeddings = samples[pos:]
    val_class_ids = class_ids[pos:]    

    train_chunks = max(train_embeddings.shape[0] // batch_size,1)
    val_chunks = max(val_embeddings.shape[0] // batch_size,1)

    train_embeddings = torch.from_numpy(train_embeddings).to(device, non_blocking=True)
    val_embeddings = torch.from_numpy(val_embeddings).to(device, non_blocking=True)
    train_class_ids = torch.from_numpy(train_class_ids).to(device, non_blocking=True)
    val_class_ids = torch.from_numpy(val_class_ids).to(device, non_blocking=True)
    
    trainset = SentenceEmbeddedDataset(train_embeddings, train_class_ids, device)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=False, num_workers=3)

    valset = SentenceEmbeddedDataset(val_embeddings, val_class_ids, device)
    
    valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size,
                                         shuffle=False, num_workers=3)

    
    train_embeddings = torch.tensor_split(train_embeddings, train_chunks)
    val_embeddings = torch.tensor_split(val_embeddings, val_chunks)
    train_class_ids = torch.tensor_split(train_class_ids, train_chunks)
    val_class_ids = torch.tensor_split(val_class_ids, val_chunks)

    best_metric = 0
    num_epoch_best_metric_unchanged = 0
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}")
        num_epoch_best_metric_unchanged += 1
        #model.eval()
        train_losses = []
        ## train loop
        time = strftime('%X', localtime())
        
        #for i, batch in enumerate(tqdm(trainloader, desc=f"{time}")):
        for i in tqdm(range(len(train_embeddings)), desc=f"{time}"):       
            embeddings, classes = train_embeddings[i], train_class_ids[i]
            optimizer.zero_grad(set_to_none=True)
            #with torch.no_grad():
            pred = model(embeddings)            
            loss_iter = criterion(pred, classes)
            loss_iter.backward()
            optimizer.step()
            train_losses.append(loss_iter)
        #print(train_losses)
        train_loss_total_norm = np.sum(train_losses) / len(trainloader)
        #print(train_loss_total_norm)

        ## validation loop
        model.eval()
        
        val_losses = []
        time = strftime('%X', localtime())
        #for i, batch in enumerate(tqdm(valloader, desc=f"{time}")):        
        for i in tqdm(range(len(val_embeddings)), desc=f"{time}"): 
            #embeddings, classes = batch
            embeddings, classes = val_embeddings[i], val_class_ids[i]
            with torch.no_grad():
                pred = model(embeddings)
            loss_iter = criterion(pred, classes)
            val_losses.append(loss_iter)
        #print(val_losses)
        val_loss_total_norm = np.sum(val_losses) / len(valloader)
        #print(val_loss_total_norm)
        print(f"Train loss: {train_loss_total_norm}\tval loss: {val_loss_total_norm}")

    return model

class TransformerEmbedder:

    def __init__(self, model_name="paraphrase-distilroberta-base-v2"):            
        self.model = SentenceTransformer(model_name)      
    
    def emb_sentence(self, sentences, normalize=False):
        x = self.model.encode(sentences)

        if not isinstance(sentences, list):            
            if normalize:
                x = x / np.linalg.norm(x, axis=0, ord=2)# + 1e-8
            return x.reshape(1,-1)
        else:
            if normalize:
                x = x / np.linalg.norm(x, axis=0, ord=2)# + 1e-8
            return x

#from data.models.jointembedding import JointEmbeddingModel

class JointEmbedder: # our model
    def __init__(self, model_path="data/models/jointembedding/20tuplas_best_model_e5",device=0):            
        jointmodel = torch.load(model_path)
        jointmodel.eval()
        self.sent_emb = jointmodel.sentence_embedding     
        self.sbert = TransformerEmbedder("paraphrase-distilroberta-base-v2")
        self.device = device
    
    def emb_sentence(self, sentences, normalize=False):
        
        x = self.sbert.emb_sentence(sentences)
        x = torch.from_numpy(np.expand_dims(np.asarray(x),axis=0)).to(f"cuda:{self.device}")
        x = self.sent_emb(x)
        x = x.data.cpu().numpy()
        return np.squeeze(x, axis=0)


#sent = ["the man is seen","the book is on the table"]
#joint = JointEmbedder()
#x = joint.emb_sentence(sent)
#print(x.shape)


def load_vid_tokens(dataset_dir="/home/valter/datasets/activitynetcaptions_features_bit", device=0, model_name="BiT-M-R152x2", model_dir="data/models/"):
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda:%d" % device)
    torch.set_grad_enabled(False)

    print("Loading model...")
    model = models.KNOWN_MODELS[model_name]()
    model.load_from(np.load(os.path.join(model_dir, f"{model_name}.npz"), allow_pickle=True))
    model = model.to(device, non_blocking=True)
    model.eval()

    print("Embedding videos...")
    files = glob.glob(dataset_dir+"/*.npy")
    vid_tokens = []
    for file in tqdm(files):
        try:    
            x = np.load(file)
            x = np.expand_dims(x, axis = -1)
            x = np.expand_dims(x, axis = -1)
            x = torch.from_numpy(x).to(device, non_blocking=True)
            logits = model.head.conv(x)[...,0,0]
            x = torch.nn.functional.softmax(logits, dim=1).data.cpu().numpy()
            obj_ids = np.argmax(x, axis=1) # apenas o objeto mais provavel                        
            vid_tokens.append(obj_ids)
                    
        except Exception as e :
            print(e)
            print(file)
            pass
    return vid_tokens


def print_confusion_matrix(y_test, y_pred, classes, w=24,h=16,d=70, show=False, save=True, absolute_values=True, file="results/cm.pdf", plot_name=""):    
    
    classes = [c.replace("_"," ") for c in list(classes.keys())]
    
    np.set_printoptions(precision=3)
    plt.figure(figsize=(w, h), dpi=d)
    data = {
        'Ocorreu': y_test,
        'Predito': y_pred
    }
    df = pd.DataFrame(data, columns=['Ocorreu','Predito'])
    if absolute_values:
        conf = pd.crosstab(df['Ocorreu'], df['Predito'], rownames=['Ocorreu'], colnames=['Predito'])
        sns_plot = sn.heatmap(conf, annot=True, fmt="d", annot_kws={"size":8}, cmap=plt.cm.Blues, xticklabels=classes, yticklabels=classes)
    else:
        conf = pd.crosstab(df['Ocorreu'], df['Predito'], rownames=['Ocorreu'], colnames=['Predito'], normalize=True)
        sns_plot = sn.heatmap(conf, annot=False, annot_kws={"size":8}, cmap=plt.cm.Blues, xticklabels=classes, yticklabels=classes)
    plt.xticks(rotation=90)
    plt.xlabel(plot_name)
    plt.ylabel("")
    plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = True
    plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = False
    if show:
        plt.show()
    if save:
        fig = sns_plot.get_figure()
        fig.savefig(file)
        #plt.savefig(file)