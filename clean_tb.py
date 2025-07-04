import argparse
import shutil

from tensorboard.backend.event_processing.event_file_inspector import get_dict_to_print, get_inspection_units

parser = argparse.ArgumentParser("delete small runs")
parser.add_argument("--logdir", type=str, default=".")
parser.add_argument("--min_run_len", type=int, default=10)
parser.add_argument("--list", action="store_true")
args = parser.parse_args()

inspect_units = get_inspection_units(logdir=args.logdir)

run_len = {}
for run in inspect_units:
    path = run.name
    max_length = 0
    for key, value in get_dict_to_print(run.field_to_obs).items():
        if value is not None:
            length = value["max_step"]
            if max_length < length:
                max_length = length
    run_len[path] = max_length

for run, length in run_len.items():
    if length < args.min_run_len:
        if args.list:
            print(f"{run} is {length} steps long and so will be deleted")
        else:
            try:
                print(f"{run} is {length} and was deleted")
                shutil.rmtree(run)
            except Exception as e:
                print(f"OS didn't let us delete {run}: {e}")
    else:
        pass
        # print(f"{run} is {length} and is good")
