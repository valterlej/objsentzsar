# HMDB51
dataset_name="hmdb"
dataset_files = "data/files/hmdb51_dataset_new.json"
#bit_predictions_dir = "data/hmdb51_bit"
#bit_predictions_dir = "../objects2action/data/hmdb51_bit_r152_2"
bit_predictions_dir = "/home/valter/Vídeos/hmdb51_bit/"
classes_descriptions = "data/texts/hmdb51_texts/"
truze_splits_file = "data/files/hmdb51_truezsl_splits.json"
random_splits = False
only_class_label = True
runs = 1
number_random_classes = 26
max_senteces = 20
min_len = 3
Tz = 20
Tv = 100
Ts = 10
Ta = 50
compute_action_sparsity = True
compute_video_sparsity = True

observers = [
    #"data/observers/ob1_hmdb51.json",
    #"data/observers/ob3_hmdb51.json"
    "data/observers/bit/hmdb51_resnet152_4096_new.json",
    "data/observers/bit/hmdb51_resnet152_4096_objectemb_768_new.json",
    "data/observers/bit/hmdb51_resnet152_objectemb_1268_new.json"
]