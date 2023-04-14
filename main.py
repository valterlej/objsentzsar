import warnings
warnings.filterwarnings("ignore")
import time
import numpy as np
import sys
from sklearn.metrics import accuracy_score
from utils import TransformerEmbedder
from dataset import load_object_predictions
from dataset import load_random_classes
from dataset import load_class_sentences
from dataset import load_trueze_classes
from dataset import load_object_classes_and_descriptions
from dataset import load_samples_from_files
from dataset import get_ids_embeddings_from_samples
from dataset import get_class_label_descriptions
from utils import scale_as_probabilities
from utils import compute_mean_desv
from utils import semantic_embedding  
from utils import compute_sparsity
from utils import get_model
from utils import print_confusion_matrix
from utils import training_deep_supervised_classifier

embedder = TransformerEmbedder(model_name="paraphrase-distilroberta-base-v2")
imagenet_file = "data/models/imagenet21k_wordnet_lemmas.txt"


#from param_ucf101 import *
#from param_hmdb51 import *
from param_kinetics400 import *

def predict_deep_supervised_szar(files, data, classes, classifier):
    sample_ids, sample_embeddings, _ = get_ids_embeddings_from_samples(files, data, classes, embedder, concat_sentences=True)
    p_sent = classifier.predict(sample_embeddings)
    return sample_ids, p_sent

def report_register(acc, text):    
    print(f"{text} {acc}")

def print_experiment_report(accs, text):
    mean, desv = compute_mean_desv(accs)
    print(f"{text}\tmean: {mean} std: {desv}")

def run_experiment(runs):    

    if not random_splits:
        runs = 1
    
    accs_objects = []
    accs_sent_supervised = []
    accs_combined_obj_sent = []

    for r in range(runs):
        
        print(80*"*")
        print(f"Summary - {r+1} of {runs}")
        print(80*"*")

        if random_splits:
            z_names = load_random_classes(truze_splits_file, number_random_classes, dataset_files)
        else:
            z_names = load_trueze_classes(truze_splits_file, "testing", dataset_files)

        sentences = load_class_sentences(classes_descriptions, embedder, min_len, max_senteces)                
                 
        z_descriptions = []
        for z_name in z_names:
            z_descriptions.append(" ".join(sentences[z_name.lower()]))
        s_z = semantic_embedding(z_descriptions, embedder) # (N_classes, 768) e.g., (34, 768)
        
        y_names, y_descriptions = load_object_classes_and_descriptions(imagenet_file)
                        
        s_y = semantic_embedding(y_descriptions, embedder) # (N_objects, 768) e.g., (21240, 768)
        data = load_samples_from_files(observers) ### sentences from observers
        
        
        files, test, p_v = load_object_predictions(bit_predictions_dir, z_names) # (N_video_samples, 21843) - equation (2)   ----- gargalo     
        
        
        
        print(p_v.shape)
        g_yz = np.matmul(s_y, s_z.T).T # obj_ac_affinity(s_y, s_z) -- (n_objects, n_classes)  -- equation (3)
        if compute_action_sparsity:
            g_yz = compute_sparsity(g_yz, Tz) # (n_classes, n_objects)
        if compute_video_sparsity:
            p_v = compute_sparsity(p_v, Tv) # (n_samples, n_objects)

        print(f"Testing set has {len(test)} samples.")

        # objects only
        #### incluindo a affinidade intra-objects computada com videos        
        #p_obj_new = np.matmul(p_v,p_obj_new.T) # (n_videos, n_classes)
        
        p_obj = np.matmul(p_v, g_yz.T) 
        preds = np.argmax(p_obj, axis=1)
        acc = 100*accuracy_score(test, preds)
        accs_objects.append(acc)
        report_register(acc, "Accuracy (objects):")

        if not random_splits:
            print_confusion_matrix(test, preds, z_names, show=False, save=True, file=f"results/cm_objects_{dataset_name}.png", plot_name="Objects")
        elif (dataset_name == "ucf" and number_random_classes == 101) or (dataset_name == "hmdb" and number_random_classes == 51):
            print_confusion_matrix(test, preds, z_names, show=False, save=True, file=f"results/cm_objects_{dataset_name}.png", plot_name="Objects")

        #### zsar based on sentences
        class_ids = []
        samples = []
        for id, z_name in enumerate(list(z_names.keys())):
            z_name = z_name.lower()
            for sent in sentences[z_name]:
                class_ids.append(id)
                samples.append(sent)
        class_ids = np.asarray(class_ids)
        samples = semantic_embedding(samples, embedder)
        
        clf = training_deep_supervised_classifier(class_ids, samples, len(list(z_names.keys())), epochs=20, batch_size=64)                
        test_ids, p_sent = predict_deep_supervised_szar(files, data, z_names, clf)
                    
        # supervised sentences
        p_sent = compute_sparsity(p_sent, Ts)
        preds = np.argmax(p_sent, axis=1)
        acc = 100*accuracy_score(test_ids, preds)
        accs_sent_supervised.append(acc)
        report_register(acc, "Accuracy (sentences -- supervised approach):")
        if not random_splits:
            print_confusion_matrix(test_ids, preds, z_names, show=False, save=True, file=f"results/cm_sent_{dataset_name}.png", plot_name="Sentences")
        elif (dataset_name == "ucf" and number_random_classes == 101) or (dataset_name == "hmdb" and number_random_classes == 51):
            print_confusion_matrix(test_ids, preds, z_names, show=False, save=True, file=f"results/cm_sent_{dataset_name}.png", plot_name="Sentences")
        
        # object + supervised sentences
        p_obj_sent = 0.6*scale_as_probabilities(p_obj) + 0.4*scale_as_probabilities(p_sent)
        preds = np.argmax(p_obj_sent, axis=1)
        acc = 100*accuracy_score(test_ids, preds)
        accs_combined_obj_sent.append(acc)
        report_register(acc, "Accuracy (combined - vidsim+sent):") 

        if not random_splits:
            print_confusion_matrix(test_ids, preds, z_names, show=False, save=True, file=f"results/cm_obj_sentences_{dataset_name}.png", plot_name="Objects + Sentences")                     
        elif (dataset_name == "ucf" and number_random_classes == 101) or (dataset_name == "hmdb" and number_random_classes == 51):
            print_confusion_matrix(test_ids, preds, z_names, show=False, save=True, file=f"results/cm_obj_sentences_{dataset_name}.png", plot_name="Objects + Sentences")                     
        
    return accs_objects, accs_sent_supervised, accs_combined_obj_sent   


start = time.time()
accs_objects, accs_sent_supervised, accs_combined_obj_sent = run_experiment(runs)

if random_splits:
    print(80*"*")
    print("Summary - Experiment")
    print(80*"*")    
    print_experiment_report(accs_objects, "Accuracy (objects)")
    print_experiment_report(accs_sent_supervised, "Accuracy (sentences -- supervised approach)")
    print_experiment_report(accs_combined_obj_sent, "Accuracy (combined - obj+sent)")    

print(f"\n\nTime taken: {time.time()-start} sec")