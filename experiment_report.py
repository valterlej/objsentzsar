import os
import json
import pickle
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from tabulate import tabulate


class ModelPredictions(object):
    def __init__(self,classes, real_ids, predictions, description=""):
        self.classes = classes
        self.real_ids = real_ids
        self.predictions = predictions
        self.description = description

class RandomRun(object):
    def __init__(self,model_predictions=[]):
        self.model_predictions = model_predictions
        self.accuracies = []
    
    def add_model_prediction(self,model_prediction):
        self.model_predictions.append(model_prediction)

    def compute_accuracy(self, top_n=1):
        pass

class Experiment(object):
    def __init__(self, random_runs=[], dataset="",dataset_classes=[],description=""):
        self.runs = random_runs
        self.dataset = dataset
        self.dataset_classes = dataset_classes
        self.description = description    

    def add_random_run(self, random_run):
        self.runs.append(random_run)

    def compute_accuracy(self, top_n=1):
        pass

    def plot_confusion_matrix(self, run, show=False, output_file=None):
        pass

    def print_report_per_class_acc_per_run(self, run=0, show_classes=False, filter_top_k=-1):
        random_run = self.runs[run]
        print("#"*80)
        print(f"Per class accuracy - {run+1} of {len(self.runs)}")
        print("#"*80)
        per_class = []
        header = ["#","Class"]
        for model_prediction in random_run.model_predictions:  
            header.append(model_prediction.description)
            preds = [np.argmax(pred,axis=1) for pred in model_prediction.predictions]
            matrix = confusion_matrix(model_prediction.real_ids, preds)
            per_class.append(matrix.diagonal()/matrix.sum(axis=1)) ### axis 0 or axis 1
        
        per_class = np.asarray(per_class)
        per_class = list(per_class.T)
        new_per_class = []
        for i, p in enumerate(per_class):
            p = list(100*p)
            p = [round(x,2) for x in p]
            p.insert(0,random_run.model_predictions[0].classes[i])
            new_per_class.append(p)
        new_per_class = sorted(new_per_class, key=lambda x: (x[len(random_run.model_predictions)],x[0]), reverse=True)        
        per_class = []
        for i, p in enumerate(new_per_class):
            p.insert(0,i+1)
            per_class.append(p)

        new_per_class = []
        if filter_top_k != -1:
            new_per_class += per_class[:filter_top_k]
            new_per_class += per_class[-filter_top_k:]
        else:
            new_per_class = per_class

        print(80*"=")
        print(tabulate(new_per_class, headers=header))    
        print(80*"=")

    def print_report_per_run(self, run=0, show_classes=False):               
        random_run = self.runs[run]
        print("#"*80)
        print(f"Report - {run+1} of {len(self.runs)}")
        print("#"*80)
        for model_prediction in random_run.model_predictions:                    
            preds = [np.argmax(pred,axis=1) for pred in model_prediction.predictions]
            acc = 100*accuracy_score(model_prediction.real_ids, preds)
            print(f"{model_prediction.description} {acc}")
        if show_classes:
            print(random_run.model_predictions[0].classes)


    def print_experiment_report(self):        
        accs = []
        model_desc = []
        print("#"*80)
        print(f"Experiment Report - {self.description}")
        print("#"*80)
        for i, random_run in enumerate(self.runs):            
            a = []
            for model_prediction in random_run.model_predictions:                    
                preds = [np.argmax(pred,axis=1) for pred in model_prediction.predictions]
                acc = 100*accuracy_score(model_prediction.real_ids, preds)
                a.append(acc)
                if len(model_desc) < len(random_run.model_predictions):
                    model_desc.append(model_prediction.description)
            accs.append(a)
        accs = np.asarray(accs)
        mean = np.mean(accs, axis=0)
        desv = np.std(accs, axis=0)
        print(80*"=")
        line = "Runs\t"
        for m in model_desc:
            line += f"{m}\t"
        print(line)
        print(80*"=")
        for i in range(accs.shape[0]):
            line = f"{i+1}\t\t"
            for j in range(len(model_desc)):
                line += f"{round(accs[i][j],2)}\t\t"
            print(line)
        print(80*"=")
        line = "Mean\t\t"
        for i in range(len(model_desc)):
            line += f"{round(mean[i],2)}\t\t"
        print(line)
        line = "Desvpad\t\t"
        for i in range(len(model_desc)):
            line += f"{round(desv[i],2)}\t\t"
        print(line)
        line = "Min acc\t\t"
        for i in range(len(model_desc)):
            line += f"{round(np.min(accs[:,i]),2)} ({np.argmin(accs[:,i])+1})\t"
        print(line)
        line = "Max acc\t\t"
        for i in range(len(model_desc)):
            line += f"{round(np.max(accs[:,i]),2)} ({np.argmax(accs[:,i])+1})\t"
        print(line)                



def load_experiment(file_name):
    with open(file_name, "rb") as file:
        return pickle.load(file)        

def save_experiment(experiment, file_name="results/ex.pkl"):
    with open(file_name, 'wb') as file:      
        pickle.dump(experiment, file)