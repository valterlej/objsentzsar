import glob
import numpy as np
import torch
import contractions
import nltk
import re
import unicodedata
import json
import sys
import random
from tqdm import tqdm
from sentence_transformers import util
from nltk.corpus import wordnet
from utils import semantic_embedding


def sample_belongs_to_class_split(classes, sample_name):
    """Returns the id for the sample if it belongs to the some class in classes

    Parameters
    ----------
    classes: list
        a list with the classes from a split (training or testing)
    sample_name: str
        sample file name
    
    Returns
    -------
    
    id int
        the class id or -1 if the sample does not belongs to the split
    """    
    id = -1
    for i, c_name in enumerate(list(classes.keys())):
        files = classes[c_name]
        for f in files:
            if f.split("/")[-1].replace(".npy","") == sample_name.split("/")[-1].replace(".npy",""):
                return i
    return -1


def load_object_predictions(bit_predictions_dir, classes):
    """Load from disk all object predictions from a directory

    Parameters
    ----------
    bit_predictions_dir: str
        a directory containing files with the logits predictions for each video
    
    Returns
    -------
    
    np.array
        an array of dimension (N_samples x 21843)
    """        
    if bit_predictions_dir[:-1] != "/":
        bit_predictions_dir += "/"
    files = []
    ids = []

    real_classes = []
    preds = []
    loaded_files = []
    for i, c in enumerate(list(classes.keys())):
        for f in classes[c]:
            files.append(f)
            ids.append(i)
    
    for i in tqdm(range(len(files))):
        id = ids[i]
        try:
            preds.append(torch.nn.functional.softmax(torch.from_numpy(np.load(bit_predictions_dir+files[i]+".npy")), dim=1).numpy())
            real_classes.append(id)
            loaded_files.append(bit_predictions_dir+files[i]+".npy")
        except:
            print(bit_predictions_dir+files[i]+".npy")

    return loaded_files, real_classes, np.asarray(preds).reshape(len(preds),-1)


def load_class_sentences(dataset_dir, embedder, min_len=10, max_sentences_per_file=10, return_json=True):   
    """Load class sentences descriptions

    Parameters
    ----------
    dataset_dir: str
        a directory containing files with all descriptions
    
    embedder: TransformerEmbedder
        an object responsible to perform Sentence Embedding with Paraphrase Pre-trained models
    
    min_len: int
        minimum length for the selected sentences
    
    max_sentences_per_file: int
        maximum number of sentences per class

    return_json: bool
        true if a json must be return
        

    Returns
    -------
    
    list
        a list with pairs of class name and sentence optionally returns in a json format

    """
    def unicode_to_ascii(s):
        return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

    
    def preprocess_sentence(w, pad_punctuation=True, only_letters_and_punctuation=True):
        w = unicode_to_ascii(w.lower().strip())
        if pad_punctuation:
            w = re.sub(r"([?.!,¿])", r" \1 ", w) # creating a space between a word and the punctuation following it
            w = re.sub(r'[" "]+', " ", w)
        if only_letters_and_punctuation:
            w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w) # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
        w = w.strip()  
        return w

    def proccess_paragraphs(lines, min_len):    
        if not isinstance(lines, list):
            lines = [lines]

        paragraphs = []
        for l in lines:
            l = l[:-1].lower().strip()
            if len(l) != 0:
                paragraphs.append(l)
        sentences = []
        for p in paragraphs:
            s = nltk.tokenize.sent_tokenize(p)
            for i in s:
                i = contractions.fix(i)
                words = i.split(" ")
                if len(words) >= min_len:
                    sentences.append(preprocess_sentence(i, pad_punctuation=True, only_letters_and_punctuation=True))
        sentences = list(set(sentences)) # remove repeated sentences
        return sentences
    
    def load_sentences(file, min_len=10):
        lines = open(file,"r", encoding="cp1251", errors='ignore').readlines()
        return proccess_paragraphs(lines, min_len) 

    def evaluate_similarity(sentences, class_embedding, embedder, max_sentences_per_file=10):   
        sentences_list = []
        embs = embedder.emb_sentence(sentences)
        cosine_scores = list(util.pytorch_cos_sim(class_embedding, embs).numpy()[0])
        sentences_list = [[x[0],x[1]] for x in zip(sentences, cosine_scores)]
        sentences_list = sorted(sentences_list, key=lambda item: -item[1])[:max_sentences_per_file]
        return [s[0] for s in sentences_list]
    
    files = glob.glob(dataset_dir+"*.txt")
    class_sentences = {}
    for file in tqdm(files):        
        class_name = file.split("/")[-1].lower().replace("_"," ")[:-4]
        class_embedding = embedder.emb_sentence(class_name)
        sentences = load_sentences(file, min_len)
        sentences = evaluate_similarity(sentences, class_embedding, embedder, max_sentences_per_file)
        if len(sentences) < max_sentences_per_file: ### showing if there are classes with less than max_sent
            print(f"{class_name} has only \t{len(sentences)} sentences.")
            sys.exit()

                
        #sentences = [class_name + " " + s for s in sentences]
        sentences = [" " + s for s in sentences]
        class_sentences[class_name] = sentences        

    if return_json:
        return class_sentences
    else:
        sentences_list = []
        for cname in list(class_sentences.keys()):
            sentences = class_sentences[cname]
            for s in sentences:
                sentences_list.append([cname, s])
        return sentences_list


def get_word_net_definition(words):
    """Return a wordnet definition for a given set of words. 
    Follows 'Elaborative Rehearsal for Zero-Shot Action Recognition'

    Parameters
    ----------
    words: list
        lista de palavras a serem procuradas na wordnet
            
    Returns
    -------
    
    str
        a paragraph with a textual definition for all the input words
    """
    return_sentence = ""
    for word in words:
        result = wordnet.synsets(word)
        if not result:
            continue        
        sentence = ""
        for item in result:
            sentence += f"{item.definition()} . "
        return_sentence += sentence
    return return_sentence


def load_object_classes_and_descriptions(file):
    """Load object classes and their corresponding descriptions

    Parameters
    ----------
    file: str
        a file with ImageNet 21k lemmas (object labels)
            
    Returns
    -------
    
    list, list
        a list with all object class names
        a list with their corresponding descriptions from word net definitions
    """    
    object_classes = [line[:-1] for line in open(file,"r").readlines()]    
    obj_desc = []
    i = 0
    for o in tqdm(object_classes):
        #obj_desc.append(" ".join(o.replace(" ","").split(",")) + get_word_net_definition(o.replace(" ","").split(",")))
        obj_desc.append(get_word_net_definition(o.replace(" ","").split(",")))
    return object_classes, obj_desc


def load_trueze_classes(file, split="testing", dataset_files=None):
    """Load a TruZe split from a json configuration file

    Parameters
    ----------
    file: str
        a json file with a class list for each split
            
    Returns
    -------
    
    list
        a list with class labels
    """
    classes = [c.lower() for c in json.loads(open(file,"r").read())[split]]
    classes_files = json.loads(open(dataset_files, "r").read())
    data = {}
    for c in classes:
        data[c] = classes_files[c.replace("_","").lower()]
    return data



def load_random_classes(file, random_classes, dataset_files):
    """Load random classes from true_ze file (using both splits)    
    
    Parameters
    ----------
    file: str
        a json file with a class list for each trueze split
    random_classes: int
        a number of random classes

    Returns
    -------
    
    list
        a list with n random class labels
    """
    tr = load_trueze_classes(file, "training", dataset_files)
    te = load_trueze_classes(file, "testing", dataset_files)
    
    classes = list(tr.keys()) + list(te.keys())    
    random.shuffle(classes)
    classes = classes[:random_classes]
    te.update(tr)
    data = {}
    for c in classes:
        data[c] = te[c]
    return data


def load_observers_data(observers):
    obs = []
    for o in observers:
        observer = json.loads(open(o).read())["results"]
        obs.append(observer)
    return obs

def load_samples_from_files(sample_files):

    data = {}

    for sample_file in sample_files:
        
        predictions = json.loads(open(sample_file).read())["results"]
        for id, sample in enumerate(list(predictions.keys())):     
            file_name = sample
            sentence = predictions[sample][0]["sentence"].lower()
            try:
                d = data[file_name]
                d["sentences"].append(sentence)
            except:
                data[file_name] = {"sentences":[]}                
                data[file_name]["sentences"].append(sentence)

    return data


def load_observers(obs_files, sample_file_names, classes):
    
    data = {}
    all_predictions = []
    for file in obs_files:        
        predictions = json.loads(open(file).read())["results"]
        all_predictions.append(predictions)

    for sample_file in tqdm(sample_file_names):
        
        
        sample_file = sample_file.split("/")[-1].replace(".npy","")
        id = sample_belongs_to_class_split(classes, sample_file)
        if id == -1:
            continue

        data[sample_file] = {"ids": [], "sentences": []}

        for prediction in predictions:
            try:
                sentence = prediction[sample_file][0]["sentence"].lower()
            except:
                sentence = classes[id].replace("_"," ").lower()
            data[sample_file]["ids"].append(id)
            data[sample_file]["sentences"].append(sentence)
            
    return data

def get_ids_embeddings_from_samples(files, data, classes, embedder, concat_sentences=True):
    zsar_test_sample_ids = []
    zsar_test_sentences = []
    for file in files:
        file = file.split("/")[-1].replace(".npy","")    
        try:
            sentences = data[file]["sentences"]
        except:
            sentences = "no sentence"
            continue # review this behavior        
        
        if concat_sentences:
            sentences = [" ".join(sentences)]
                        
        id = sample_belongs_to_class_split(classes, file)
        if id != -1:
            for sentence in sentences:
                zsar_test_sample_ids.append(id)
                zsar_test_sentences.append(sentence)
    zsar_test_embeddings = semantic_embedding(zsar_test_sentences, embedder)
    return zsar_test_sample_ids, zsar_test_embeddings, zsar_test_sentences


def get_class_label_descriptions(z_names, elab_descriptions, embedder, only_label=False):

    descriptions = json.loads(open(elab_descriptions,"r").read())

    class_descriptions = []
    for class_name in list(z_names.keys()):
        for desc in descriptions:
            if class_name.replace(" ","").replace("_","").lower() == desc["word"].replace(" ","").replace("_","").lower():
                l = desc["word"]
                d = desc["cleaned_defn"]
                if not only_label:                    
                    class_descriptions.append(f"{l} {d}")
                else:
                    class_descriptions.append(f"{l}")
                break
      
    if len(class_descriptions) != len(z_names):
        print(len(class_descriptions))
        print(len(z_names))
        sys.exit("Invalid descriptions")
    
    return semantic_embedding(class_descriptions, embedder)


from collections import Counter, defaultdict


class CoOcurrenceEstimation:

    def __init__(self, vid_tokens, window_size=25, vocab_size=21843):
        self._window_size = window_size
        self.tokens = vid_tokens
        self.vocab_size = vocab_size
        self._word2id = {w:i for i, w in enumerate(range(0,vocab_size))}
        self._id2word = {i:w for w, i in self._word2id.items()}
        self._vocab_len = len(self._word2id)
        self._ids_tokens = []
        for sent_tokens in self.tokens:
                self._ids_tokens.append([self._word2id[w] for w in sent_tokens])
        self.coocurrence_matrix = self._create_coocurrence_matrix()       

    def _create_coocurrence_matrix(self):
        cooc_mat = defaultdict(Counter)        
        
        for _id_tokens in self._ids_tokens:        
            for i, w in enumerate(_id_tokens):
                start_i = max(i - self._window_size, 0)
                end_i = min(i + self._window_size + 1, len(_id_tokens))
                for j in range(start_i, end_i):
                    if i != j:
                        c = _id_tokens[j]
                        cooc_mat[w][c] += 1 / max(abs(j-1), 1)                
        
        
        mat = np.zeros((self.vocab_size,self.vocab_size), dtype=np.float32)

        for w, cnt in cooc_mat.items(): #3412: Counter({2167: 5.83, 90: 0.11, 19: 0.08, 2322: 0.03, 94: 0.03, 21713: 0.03})
            for c, v in cnt.items():
                mat[w,c] = v        
        return mat

        """
        self._i_idx = list()
        self._j_idx = list()
        self._xij = list()

        # create indexes and x value tensors
        for w, cnt in cooc_mat.items():
            for c, v in cnt.items():
                self._i_idx.append(w)
                self._j_idx.append(c)
                self._xij.append(v)
        
        self._i_idx = torch.LongTensor(self._i_idx).cuda()
        self._j_idx = torch.LongTensor(self._j_idx).cuda()
        self._xij = torch.FloatTensor(self._xij).cuda()
        """