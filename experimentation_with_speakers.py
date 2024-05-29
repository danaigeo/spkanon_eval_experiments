import os
import pickle
import json

import random

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import roc_curve, RocCurveDisplay
import numpy as np
import plda
from omegaconf import OmegaConf
from matplotlib import pyplot as plt
import torch
from argparse import ArgumentParser

from spkanon_eval.setup_module import setup
from spkanon_eval.datamodules.dataset import load_audio
from spkanon_eval.evaluation.asv.asv_utils import analyse_results, compute_eer, compute_llrs
from spkanon_eval.evaluation.asv.spkid_plda import ASV
from spkanon_eval.evaluation.analysis import get_characteristics

def sample_and_filter_speakers(args,write_file, start_speakers, end_speakers, increment_speakers, iterations_per_group, seed=None):
    eval_folder =args.config
    scenario = args.scen

    asv_folder = os.path.join("spkanon_eval/logs/stargan/train/default-360/1711973011/eval/asv-plda", scenario, "train")
    trial_folder = os.path.join(eval_folder, "results", "eval")
    enroll_folder =  os.path.join(eval_folder, "results", "eval_enrolls")
    sample_rate = 16000

    lda_model = pickle.load(open(os.path.join(asv_folder, "models", "lda.pkl"), "rb"))
    plda_model = pickle.load(open(os.path.join(asv_folder, "models", "plda.pkl"), "rb"))

    spkemb_config = OmegaConf.load("spkanon_eval/config/components/spkid/xvector.yaml")
    spkemb_config.spkid.emb_model_ckpt = None
    spkemb_model = setup(spkemb_config, "cpu")["spkid"]

    spkemb_config.spkid.batch_size = 24
    asv_config = OmegaConf.create(
        {
            "scenario": scenario,
            "spkid": spkemb_config.spkid,
            "lda_ckpt": os.path.join(asv_folder, "models", "lda.pkl"),
            "plda_ckpt": os.path.join(asv_folder, "models", "plda.pkl"),
            "data": {
                "config": {
                    "batch_size": 2,
                    "num_workers": 10,
                    "sample_rate": 16000,
                }
            }
        }
    )
    asv = ASV(asv_config, "cpu", torch.nn.Linear(256, 256))
    f_trials = os.path.join(eval_folder, "data", "eval_trials.txt")
    if scenario == "ignorant":
        f_enrolls = os.path.join(eval_folder, "data", "eval_enrolls.txt")
    else:
        f_enrolls = os.path.join(eval_folder, "data", "anon_eval_enrolls.txt")


    seed_eval = args.seed_eval
    with open(write_file, "a") as f:
        f.write(f"scenario seed file num_of_speakers unique_pairs threshold eer\n")


    # Optional: Set a seed for reproducibility
    if seed is not None:
        random.seed(seed)

    # Step 1: Collect all unique spe
    new_df = f_trials
    # Collect all unique speaker_ids
    all_speaker_ids = set()
    with open(new_df, 'r') as file:
        for line in file:
            obj = json.loads(line.strip())
            all_speaker_ids.add(obj["speaker_id"])

    # #Choose min(max_speakers, len(all_speaker_ids)) random speaker_ids
    # Generate speakers_list based on the provided start, end, and increment values
    speakers_list = list(range(start_speakers, end_speakers + 1, increment_speakers))

    for max_speakers in speakers_list:
        with open(write_file, "a") as f:
                    f.write(f"\n")
        for i in range(iterations_per_group):
            # Step 2: Choose min(max_speakers, len(all_speaker_ids)) random speaker_ids
            chosen_ids = random.sample(list(all_speaker_ids), min(max_speakers, len(all_speaker_ids)))
            print(f"Group size {max_speakers}, Iteration {i+1}, Chosen speakers: {chosen_ids}")

            trials_ids = []
            with open(f_trials, 'r') as trial_file:
                for line in trial_file:
                    obj = json.loads(line.strip())
                    if obj["speaker_id"] in chosen_ids:
                        trials_ids.append(obj)
            filtered_trials = f"trials_ig{seed_eval}_{args.end}.txt"
            with open(filtered_trials, 'w') as temp_file:
                for entry in trials_ids:
                    temp_file.write(json.dumps(entry) + "\n")
                    
            enrolls_ids = []
            with open(f_enrolls, 'r') as enroll_file:
                for line in enroll_file:
                    obj = json.loads(line.strip())
                    if obj["speaker_id"]  in chosen_ids:
                        enrolls_ids.append(obj)
        
                filtered_enrolls = f"enrolls_ig{seed_eval}_{args.end}.txt"
                with open(filtered_enrolls, 'w') as temp_file:
                    for entry in enrolls_ids:
                        temp_file.write(json.dumps(entry) + "\n")
        
                # compute SpkId vectors of all utts and map them to PLDA space
                vecs, labels = dict(), dict()
                for name, f in zip(["trials", "enrolls"], [filtered_trials, filtered_enrolls]):
                    vecs[name], labels[name] = asv.compute_spkid_vecs(f)
                    vecs[name] -= np.mean(vecs[name], axis=0)
                    if asv.lda_model is not None:
                        vecs[name] = asv.lda_model.transform(vecs[name])
                    vecs[name] = asv.plda_model.model.transform(
                        vecs[name], from_space="D", to_space="U_model"
                    )
                
                # compute LLRs of all pairs of trial and enrollment utterances
                llrs, pairs = compute_llrs(asv.plda_model, vecs, 5000)
                del vecs
                
                # map utt indices to speaker indices
                pairs[:, 0] = labels["trials"][pairs[:, 0]]
                pairs[:, 1] = labels["enrolls"][pairs[:, 1]]
                
                print("Averaging LLRs across speakers")
                print(f"No. of speaker pairs: {pairs.shape[0]}")
                
                # avg. LLRs across speakers and dump them to the experiment folder
                unique_pairs, inverse = np.unique(pairs, axis=0, return_inverse=True)
                llr_avgs = np.bincount(inverse, weights=llrs) / np.bincount(inverse)
            
                trials = unique_pairs[:, 0]
                enrolls =  unique_pairs[:, 1]  
                
                fpr, tpr, thresholds, key = compute_eer(trials, enrolls, llr_avgs)
                eer = (fpr[key] + (1 - tpr[key])) / 2
                print( llr_avgs.size, thresholds[key], eer)
                
                with open(write_file, "a") as f:
                    f.write(f"{scenario} {seed} {seed_eval} {max_speakers} {llr_avgs.size} {thresholds[key]} {eer}\n")
                    
def main(args):
    """
    1. Select the .txt with the experiments.
    2. Plot the EERs of the ignorant and lazy-informed scenario.
    """

    start_speakers = args.start  # Starting number of speaker IDs
    end_speakers = args.end  # Ending number of speaker IDs
    increment_speakers = 50 # Increment number of speaker IDs for each step
    iterations_per_group = 5  # Number of times you want to repeat the sampling process for each group size
    seed = args.seed # Seed for random number generation for reproducibility
    write_file =args.eval_file

    sample_and_filter_speakers(args, write_file, start_speakers, end_speakers, increment_speakers, iterations_per_group, seed)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to eval folder")
    parser.add_argument("--eval_file", type=str, help="Path to eval file", default= "spkanon_eval/results_stargan/ls-train-other-500/speakers_ls_train-other-500#0-lz.txt")
    parser.add_argument("--scen", type=str, help="Which scenario", default ="ignorant")
    parser.add_argument("--seed", type=int, help="Seed index", default=52)
    parser.add_argument("--seed_eval", type=int, help="Seed index", default="800")
    parser.add_argument("--start", type=int, help="Start speaker index", default=650)
    parser.add_argument("--end", type=int, help="End speaker index", default=1050)
    
    args = parser.parse_args()
    main(args)