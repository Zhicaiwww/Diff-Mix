import sys
import os
sys.path.append('/data/zhicai/code/da-fusion/')
import time
from collections import defaultdict
import re
import os
import pandas as pd
from datasets import load_dataset, load_from_disk, ClassLabel, Sequence
# ds_train, ds_val = get_split_pet_v2()
ds = load_dataset('/data/zhicai/cache/huggingface/datasets/pcuenq___oxford-pets/pcuenq--oxford-pets-3a93ac9803e6e9c9',split='train')
ds = ds.cast_column("label", Sequence(ClassLabel(names='label')))
# ds = load_from_disk('/data/zhicai/cache/huggingface/local/pet', 
print('h')
# ds.save_to_disk()
