# UCF101
dataset_name="ucf"
dataset_files = "data/files/ucf101_dataset.json"
#bit_predictions_dir = "data/ucf101_bit"
#bit_predictions_dir = "data/ucf101_bit_r152_2/"
#bit_predictions_dir = "/media/valter/experiments/bit/ucf101/ucf101_bit_152/"
bit_predictions_dir = "/home/valter/datasets/bit/ucf101/ucf101_bit_152/"
classes_descriptions = "data/texts/ucf101_texts/"
truze_splits_file = "data/files/ucf101_truezsl_splits.json"
random_splits = True
runs = 1
number_random_classes = 101 #50
max_senteces = 22 # 15
min_len = 3 # 5
Tz = 20
Tv = 100 # esparcidade para as predicoes de objetos
Ts = 10 # esparcidade para as probabilidades do modelo supervisionado
Ta = 20 # esparcidade para o vetor de afinidade # (n_classes, all_semantic_sentences)
compute_action_sparsity = True
compute_video_sparsity = True
only_class_label = True # use only label name for action class embedding

observers = [
    #"data/observers/ob1_ucf101.json",    
    #"data/observers/ucf101_bit_obj_embedding_v2_observer.json",
    #"data/observers/ob2_ucf101.json",
    #"data/observers/ob3_ucf101.json",
    #"data/observers/ob4_ucf101.json",
    #"data/observers/ob5_ucf101.json"    
    #"data/observers/ucf101_bit_d20.json",
    #"data/bservers/ucf101_results_val_pred_prop_e87_best_bit.json"
    "data/observers/bit/ucf101_resnet152_4096.json", #--- usar este 0314213539 --- considera as features de 4096d
    "data/observers/bit/ucf101_resnet_objectemb_768.json", ### usar este 0326190517 --- considera as features de 500d
    "data/observers/bit/ucf101_resnet152_objectemb_1268.json" ## usar este 0405170058 -- considera as features de 1268d
]



### pensar em melhorias para o modelo de selecao de pares objeto-sentenca
# python extract_bit_feature.py --model BiT-M-R152x2 --model_dir data/models/ --video_dir /home/valter/datasets/hmdb51_mp4_grouped/ --video_meta_file /home/valter/hmdb.jsonl --output_dir /home/valter/datasets/hmdb_bit_r152_2 --num_workers 3 
# python extract_bit_feature.py --model BiT-M-R152x2 --model_dir data/models/ --video_dir /home/valter/datasets/UCF-101_mp4/ --video_meta_file /home/valter/ucf.jsonl --output_dir /home/valter/datasets/ucf_bit_r152_2 --num_workers 3
