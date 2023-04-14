# Kinetisc 400
dataset_name="kinetics"
dataset_files = "data/files/kinetics400_dataset.json"
#bit_predictions_dir = "/media/valter/experiments/bit/kinetics400/kinetics_bit/"
bit_predictions_dir = "/home/valter/datasets/bit/kinetics400/kinetics_bit/"
classes_descriptions = "data/texts/kinetics400_texts/"
truze_splits_file = "data/files/kinetics400_truezsl_splits.json"
random_splits = True
runs = 1
number_random_classes = 100#400
max_senteces = 20
min_len = 3
Tz = 20
Tv = 100 # esparcidade para as predicoes de objetos
Ts = 10 # esparcidade para as probabilidades do modelo supervisionado
Ta = 20 # esparcidade para o vetor de afinidade # (n_classes, all_semantic_sentences)
compute_action_sparsity = True
compute_video_sparsity = True
only_class_label = True # use only label name for action class embedding

observers = [
    "data/observers/bit/kinetics400_resnet152_4096.json", #--- usar este 0314213539 --- considera as features de 4096d
    "data/observers/bit/kinetics400_resnet_objectemb_768.json", ### usar este 0326190517 --- considera as features de 500d
    "data/observers/bit/kinetics400_resnet152_objectemb_1268.json" ## usar este 0405170058 -- considera as features de 1268d
]
