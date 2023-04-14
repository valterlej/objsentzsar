#!/usr/bin/bash
from contextlib import redirect_stderr
import warnings

from pkg_resources import working_set
warnings.filterwarnings("ignore")
import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import numpy as np
import pickle
from sklearn.metrics import accuracy_score
from utils import TransformerEmbedder, training_deep_supervised_classifier_pytorch, JointEmbedder
from dataset import load_observers_data
from dataset import load_random_classes
from dataset import load_class_sentences
from dataset import load_trueze_classes
from dataset import load_object_classes_and_descriptions
from utils import scale_as_probabilities
from utils import compute_mean_desv
from utils import semantic_embedding  
from utils import compute_sparsity
from utils import print_confusion_matrix
import time
from progressbar import ProgressBar
import numpy as np
import torch
import torch.multiprocessing as mp
from utils import scale_as_probabilities
from experiment_report import RandomRun, ModelPredictions, Experiment, save_experiment, load_experiment


from param_ucf101 import *
#from param_hmdb51 import *
#from param_kinetics400 import *

embedder = TransformerEmbedder(model_name="paraphrase-distilroberta-base-v2")
jointembedder = JointEmbedder("data/models/jointembedding/20tuplas_best_model_e5")
imagenet_file = "data/models/imagenet21k_wordnet_lemmas.txt"

experiment_report_list = []

def write_report():
    f = open("./results/report_joint_embedding.txt", "w")
    for line in experiment_report_list:
        f.write(line+"\n")
    f.close()

class VideoDataset(torch.utils.data.Dataset):
    def __init__(self, directory, files, ids, observers, classes):
        super().__init__()
        self.directory = directory
        self.files = files
        self.ids = ids
        self.observers = observers
        self.classes = classes
        self.sentence_classifier = torch.load("data/tmp/sentencemodel.pt")
        self.sentence_classifier.eval()
        self.sentence_classifier = torch.nn.DataParallel(self.sentence_classifier) 
        self.affinity_matrix = np.load("data/tmp/affinity.npy")
        self.device = torch.device("cuda:%d" % 0)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        
        try:
            feature_file = os.path.join(self.directory, self.files[idx]+".npy")
            pred = torch.nn.functional.softmax(torch.from_numpy(np.load(feature_file)), dim=1).numpy()      
            pred = compute_sparsity(pred, 100)
            id = self.ids[idx]
            p_obj = np.matmul(pred, self.affinity_matrix.T) 
            object_pred = np.argmax(p_obj, axis=1)           
            sentences = []
            for observer in self.observers:
                sentences.append(observer[self.files[idx]][0]["sentence"])
            sentences = " ".join(sentences)
            logits = self.sentence_classifier(torch.from_numpy(embedder.emb_sentence(sentences.lower(), normalize=False)).to(self.device, non_blocking=True))
            p_sent = torch.nn.functional.softmax(logits, dim=1).data.cpu().numpy()
            p_sent = compute_sparsity(p_sent, 10) # sparcity of sentence predictions
            sentence_pred = np.argmax(p_sent, axis=1)                        
            
            
            p_obj_sent = (0.6*scale_as_probabilities(p_obj) + 0.4*scale_as_probabilities(p_sent))
            object_sentence_pred = np.argmax(p_obj_sent, axis=1)
            #return self.files[idx], id, object_pred, sentence_pred, object_sentence_pred
            #return self.files[idx], id, object_pred, sentence_pred, object_sentence_pred
            return self.files[idx], id, p_obj, p_sent, p_obj_sent

        except Exception as e:
            print(self.files[idx])
            print(e)
            return "", [], [], [], []


def compute_predictions(proc_id, log_queue, device, num_data_workers, directory, files, ids, observers, classes):
    print(f"Process{proc_id} starts to predict using GPU{device}")

    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda:%d" % device)
    torch.set_grad_enabled(False)

    real_class_list = []
    object_pred_list = []
    sentence_pred_list = []
    object_sentence_pred_list = []

    print(f"\tProcess{proc_id}:")

    video_dataset = VideoDataset(
      directory, files, ids, observers, classes)    
  
    video_loader = torch.utils.data.DataLoader(
      video_dataset, batch_size=8, shuffle=False, 
      drop_last=False, num_workers=num_data_workers, pin_memory=True)
  
    for name, real_class, object_pred, sentence_pred, object_sentence_pred in video_loader:
        real_class_list += [x for x in real_class.numpy()]
        object_pred_list += [x for x in object_pred.numpy()]
        sentence_pred_list += [x for x in sentence_pred.numpy()]
        object_sentence_pred_list += [x for x in object_sentence_pred.numpy()]
        log_queue.put(name)

   
    file_name = "data/tmp/workers.pkl"
    if not os.path.isfile(file_name):
        worker_data = [[],[],[],[]]
        open_file = open(file_name, "wb")
        pickle.dump(worker_data, open_file)
        open_file.close()
    
    open_file = open(file_name, "rb")
    worker_data = pickle.load(open_file)
    open_file.close()
    worker_data[0] += real_class_list
    worker_data[1] += object_pred_list
    worker_data[2] += sentence_pred_list
    worker_data[3] += object_sentence_pred_list
    open_file = open(file_name, "wb")
    pickle.dump(worker_data, open_file)
    open_file.close()
    
    #acc = 100*accuracy_score(real_class_list, object_pred_list)
    #report_register(acc, "Accuracy (objects):")
    #acc = 100*accuracy_score(real_class_list, sentence_pred_list)
    #report_register(acc, "Accuracy (sentences):")
    #acc = 100*accuracy_score(real_class_list, object_sentence_pred_list)
    #report_register(acc, "Accuracy (object + sentences):")
    log_queue.put(None)




def report_register(acc, text):    
    print(f"{text} {acc}")    

def print_experiment_report(accs, text):
    mean, desv = compute_mean_desv(accs)
    print(f"{text}\tmean: {mean} std: {desv}")
    experiment_report_list.append(f"{text}\tmean: {mean} std: {desv}")




def main():

    experiment = Experiment(dataset="ucf101",dataset_classes=[],description="kinetics400_0-400_1run")

    mp.set_start_method('spawn', force=True)
    if not random_splits:
        runs = 1
    
    accs_objects = []
    accs_sent_supervised = []
    accs_combined_obj_sent = []
    runs = 50
    num_workers = 2
    for r in range(runs):
        
        print(80*"*")
        print(f"Summary - {r+1} of {runs}")
        experiment_report_list.append(f"Summary - {r+1} of {runs}")
        print(80*"*")

        if random_splits:
            z_names = load_random_classes(truze_splits_file, number_random_classes, dataset_files)
        else:
            z_names = load_trueze_classes(truze_splits_file, "testing", dataset_files)

        sentences = load_class_sentences(classes_descriptions, embedder, min_len, max_senteces)                
        

        # obter as sentencas descritivas das classes e organizá-las por conceitos aprendidos por agrupamento
        ac_descs = []
        for z_name in z_names:
            for sent in sentences[z_name.lower()][:15]:
                ac_descs.append(sent)
        
      

        z_descriptions = []
        for z_name in z_names:
            z_descriptions.append(" ".join(sentences[z_name.lower()][:15])) ##### tomando apenas as 5 sentencas com maior similaridade e nao todas as 15

        s_z = semantic_embedding(z_descriptions, embedder) # (N_classes, 768) e.g., (34, 768)
        
        y_names, y_descriptions = load_object_classes_and_descriptions(imagenet_file)
                        
        #s_y = semantic_embedding(y_descriptions, embedder) # (N_objects, 768) e.g., (21240, 768)
        s_y = semantic_embedding(y_names, embedder)
        
        
        #data = load_samples_from_files(observers) ### sentences from observers
        data = load_observers_data(observers)                

        print("Learning object-class affinities...")
        ##################files, test, p_v = load_object_predictions(bit_predictions_dir, z_names) # (N_video_samples, 21843) - equation (2)   ----- gargalo       
        ### object classification
        g_yz = np.matmul(s_y, s_z.T).T # obj_ac_affinity(s_y, s_z) -- (n_objects, n_classes)  -- equation (3)
        g_yz = compute_sparsity(g_yz, 20) # (n_classes, n_objects) ### affinity sparcity
        #if compute_video_sparsity:
        #    p_v = compute_sparsity(p_v, Tv) # (n_samples, n_objects)
        np.save("data/tmp/affinity.npy", g_yz)

        print("Learning sentence classifier...")
        #### sentence classifier
        class_ids = []
        samples = []
        for id, z_name in enumerate(list(z_names.keys())):
            z_name = z_name.lower()
            for sent in sentences[z_name]:
                class_ids.append(id)
                samples.append(sent)
        class_ids = np.asarray(class_ids)
        #samples = semantic_embedding(samples, embedder)
        samples = semantic_embedding(samples, jointembedder)

        #clf = training_deep_supervised_classifier(class_ids, samples, len(list(z_names.keys())), epochs=20, batch_size=64)                
        #clf.save("data/tmp/sentencemodel.keras")
        
        clf = training_deep_supervised_classifier_pytorch(class_ids, samples, len(list(z_names.keys())), epochs=30, batch_size=256, hidden_size=64, drop_rate=0.1)
        torch.save(clf,"data/tmp/sentencemodel.pt")

        ########
        #  LOADING DATA
        ########
        files = []
        ids = []
        for i, c in enumerate(list(z_names.keys())):
            for f in z_names[c]:
                files.append(f)
                ids.append(i)

        print('total videos: %d' % (len(files)))
        if len(files) > 0:            
            log_queue = mp.Queue()

            num_workers = min(len(files), num_workers)
            avg_videos_per_worker = len(files) // num_workers
            num_gpus = 1#torch.cuda.device_count()
            assert num_gpus > 0, 'No GPU available'

            processes = []
            for i in range(num_workers):
                sidx = avg_videos_per_worker * i
                eidx = None if i == num_workers - 1 else sidx + avg_videos_per_worker
                device = i % num_gpus

                process = mp.Process(
                target=compute_predictions, args=(i, log_queue, device, num_workers, bit_predictions_dir, files[sidx: eidx], ids[sidx: eidx], data, z_names)
                )                       
                process.start()
                processes.append(process)

            progress_bar = ProgressBar(max_value=len(files) // (num_workers))
            progress_bar.start()

            num_finished_workers, num_finished_files = 0, 0
            while num_finished_workers < num_workers:
                res = log_queue.get()
                if res is None:
                    num_finished_workers += 1
                else:
                    num_finished_files += 1
                    progress_bar.update(num_finished_files)

            progress_bar.finish()

            for i in range(num_workers):
                processes[i].join()


            real_class_list = []
            object_pred_list = []
            sentence_pred_list = []
            object_sentence_pred_list = []
            ## carregar valores dos arquivos
            file_name = "data/tmp/workers.pkl"
            open_file = open(file_name, "rb")
            worker_data = pickle.load(open_file)
            open_file.close()
            real_class_list = worker_data[0]
            object_pred_list  = worker_data[1]
            sentence_pred_list = worker_data[2]
            object_sentence_pred_list = worker_data[3]
            
            objectModel = ModelPredictions(list(z_names.keys()),real_class_list,object_pred_list,"objects")
            sentenceModel = ModelPredictions(list(z_names.keys()),real_class_list,sentence_pred_list,"sentences")
            objectSentenceModel = ModelPredictions(list(z_names.keys()),real_class_list,object_sentence_pred_list,"objects and sentences")            
            experiment.add_random_run(RandomRun([objectModel, sentenceModel, objectSentenceModel]))
            save_experiment(experiment, file_name="results/experiment_ucf101_jointembedding.pkl")
            #experiment.save("results/experiment_test.pkl")
            
            os.system(f"rm {file_name}")

            print("#"*80)
            print(f"Report - {r+1} of {runs}")
            print("#"*80)
            ### mostrar os resultados
            # objects
            
            preds = [np.argmax(obj,axis=1) for obj in object_pred_list]
            acc = 100*accuracy_score(real_class_list, preds)
            accs_objects.append(acc)
            report_register(acc, "Accuracy (objects):")
            experiment_report_list.append(f"Objects: {acc}")

            #if not random_splits:
            #    print_confusion_matrix(real_class_list, object_pred_list, z_names, show=False, save=True, file=f"results/cm_objects_{dataset_name}.png", plot_name="Objects")
            #elif (dataset_name == "ucf" and number_random_classes == 101) or (dataset_name == "hmdb" and number_random_classes == 51):
            #    print_confusion_matrix(real_class_list, object_pred_list, z_names, show=False, save=True, file=f"results/cm_objects_{dataset_name}.png", plot_name="Objects")

            #sentences
            preds = [np.argmax(sent,axis=1) for sent in sentence_pred_list]
            acc = 100*accuracy_score(real_class_list, preds)            
            accs_sent_supervised.append(acc)
            report_register(acc, "Accuracy (sentences -- supervised approach):")
            experiment_report_list.append(f"Sentences: {acc}")
            #if not random_splits:
            #    print_confusion_matrix(real_class_list, sentence_pred_list, z_names, show=False, save=True, file=f"results/cm_sent_{dataset_name}.png", plot_name="Sentences")
            #elif (dataset_name == "ucf" and number_random_classes == 101) or (dataset_name == "hmdb" and number_random_classes == 51):
            #    print_confusion_matrix(real_class_list, sentence_pred_list, z_names, show=False, save=True, file=f"results/cm_sent_{dataset_name}.png", plot_name="Sentences")

            # combined
            # object + supervised sentences
            preds = [np.argmax(obj_sent,axis=1) for obj_sent in object_sentence_pred_list]
            acc = 100*accuracy_score(real_class_list, preds)       
            accs_combined_obj_sent.append(acc)
            report_register(acc, "Accuracy (combined - vidsim+sent):") 
            experiment_report_list.append(f"Combined: {acc}")

                        
            #print_confusion_matrix(real_class_list, [o[0] for o in object_sentence_pred_list], z_names, show=False, save=True, file=f"results/cm_obj_sentences_{dataset_name}_{len(z_names)}.png", plot_name="Objects + Sentences")                     
            
            write_report()
            #if not random_splits:
            #    print_confusion_matrix(real_class_list, object_sentence_pred_list, z_names, show=False, save=True, file=f"results/cm_obj_sentences_{dataset_name}.png", plot_name="Objects + Sentences")                     
            #elif (dataset_name == "ucf" and number_random_classes == 101) or (dataset_name == "hmdb" and number_random_classes == 51):
            #    print_confusion_matrix(real_class_list, object_sentence_pred_list, z_names, show=False, save=True, file=f"results/cm_obj_sentences_{dataset_name}.png", plot_name="Objects + Sentences")                     
    
    
    #random_splits = True
    #if random_splits:
    print(80*"*")
    print("Summary - Experiment")
    experiment_report_list.append(f"Summary experiment:")
    print(80*"*")    
    print_experiment_report(accs_objects, "Accuracy (objects)")    
    print_experiment_report(accs_sent_supervised, "Accuracy (sentences -- supervised approach)")
    print_experiment_report(accs_combined_obj_sent, "Accuracy (combined - obj+sent)")    
    write_report()        

if __name__ == "__main__":  
  start = time.time()
  main()  
  print(f"\n\nTime taken: {time.time()-start} sec")