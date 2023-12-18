from datasets import load_dataset
import logging
import time
from multiprocessing import Process
from proxy import set_proxy

def thread_function(name, path, kwargs):
    logging.info(f"Thread {name}: starting download {path} from hub")
    load_dataset(path,**kwargs)
    logging.info(f"Thread {name}: finishing download {path}")

if __name__ == "__main__":
    set_proxy()

    format = "%(asctime)s: %(message)s"
    DATASETS={
            # 'keremberke/chest-xray-classification':{'name':'full'},
            #   'Multimodal-Fatima/StanfordCars_train':{},
            #   'Multimodal-Fatima/Food101_test':{},
                'Multimodal-Fatima/Food101_test':{},
                'Multimodal-Fatima/Food101_train':{},

    } 

    logging.basicConfig(format=format, level=logging.INFO,datefmt="%H:%M:%S")
    jobs = list()
    for index, key in enumerate(DATASETS.keys()):
        x = Process(target=thread_function, args=(index, key, DATASETS[key]))
        jobs.append(x)
        x.start() 
    for index, thread in enumerate(jobs):
        thread.join()
