#
from datasets import load_dataset

NUM_PROCS = 24

# # load fineweb-edu with parallel processing
# dataset = load_dataset("HuggingFaceFW/fineweb-edu", name="default", num_proc=NUM_PROCS) # , cache_dir="/your/cache/path")

# or load a subset with roughly 100B tokens, suitable for small- or medium-sized experiments
dataset = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-100BT", num_proc=NUM_PROCS) # , cache_dir="/your/cache/path")))
