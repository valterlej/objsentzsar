import json
import os
import glob
from tqdm import tqdm
import torch.multiprocessing as mp
from progressbar import ProgressBar


data_k400 = json.loads(open("data/kinetics400/dataset/validate.json","r").read())

def download(ids, videos_target, log_queue):
    for youtube_id in tqdm(ids):
        x = data_k400[youtube_id]
        start = x["annotations"]["segment"][0]
        end = x["annotations"]["segment"][1]
        #youtube_link = "https://www.youtube.com/watch?v="+youtube_id[2:] # acnet
        youtube_link = "https://www.youtube.com/watch?v="+youtube_id[:] # kinetics     
        vid_path = os.path.join(videos_target, "v_"+youtube_id+".mp4")       
        command_acnet = "ffmpeg -loglevel panic -ss 0 -i $(youtube-dl -f 18 --get-url "+youtube_link+") -c:v copy -c:a copy '"+vid_path+"'"
        command_kinetics = "ffmpeg -loglevel panic -y -ss "+str(start)+" -to "+str(end)+" -i $(youtube-dl -f 18 --get-url "+youtube_link+") -c:v copy -c:a copy '"+vid_path+"'"
        os.system(command_kinetics)
        log_queue.put(youtube_id)
    log_queue.put(None)


def main(todo_video_list, videos_target, num_workers):

    mp.set_start_method('spawn')
    log_queue = mp.Queue()

    num_workers = min(len(todo_video_list), num_workers)
    avg_videos_per_worker = len(todo_video_list) // num_workers
    num_gpus = 1#torch.cuda.device_count()
    assert num_gpus > 0, 'No GPU available'

    processes = []
    for i in range(num_workers):
      sidx = avg_videos_per_worker * i
      eidx = None if i == num_workers - 1 else sidx + avg_videos_per_worker
      device = i % num_gpus

      process = mp.Process(
        target=download, args=(todo_video_list[sidx: eidx], videos_target, log_queue)
      )
      process.start()
      processes.append(process)

    progress_bar = ProgressBar(max_value=len(todo_video_list))
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


    print("Finish")

if __name__ == "__main__":
    ### acnet
    #ids_file = "/home/valter/Documentos/new_projects/CustomBMT/data/val_1.json"   
    #ids_file = "/media/valter/Arquivos/CustomBMT/data/train.json"
    #videos_target = "/home/valter/datasets/activitynetcaptions/"

    # kinetics400
    ids_file = "data/kinetics400/dataset/validate.json"
    videos_target = "/home/valter/datasets/kinetics400videos/"
    
    
    ids = json.loads(open(ids_file,"r").read())
    ids = list(ids.keys())
    ids = ids[:]


    files = glob.glob(videos_target+"*.mp4")
    d_ids = [f.split("/")[-1][:-4][2:] for f in files]
    #from tqdm import tqdm
    for i, id in enumerate(d_ids):
        if (i+1) % 500 == 0:
          print(f"{i+1}")
        try:
            ids.remove(id)    
        except:
            pass
    print(len(ids))
    main(ids, videos_target, 20)


# use youtube-dl and ffmpeg to download videos
# use quotation to cater for special charaters such as whitesspace and () in file or folder name
# REF: https://stackoverflow.com/q/22766111/3901871
# REF: zsnhttps://ffmpeg.org/ffmpeg-utils.html#toc-Examples
