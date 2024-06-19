import os
import pickle
import json

import random
import time

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

def sample_and_filter_speakers(args, write_file, start_speakers, end_speakers, increment_speakers, iterations_per_group, seed=None):
    """
    Function to perform ASV evaluation on the filtered eval files

    1. Loads model
    2. Resamples multiple times for each subset
    3. Performs ASV evaluation on filtered trial and enrollment evaluation files
    4. Computes EER
    5. Dumps results in the write_file

    Parameters:
    write_file: Path to file where results are dumped
    start_speakers: Lower bound for incrementation
    end_speakers: Upper bound for incrementation
    increment_speaker: Number to increment the speakers
    iterations_per_group: Defines the amount of resampling per group
    original_enrolls: Path to original enroll file (either anonymized or non-anonymized)
    num_speakers: Amount of filtered speakers
    """
    eval_folder =args.config
    scenario = args.scen

    # Load model
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
    # If the scenario is "lazy-informed", use th anonymized enrollment data

    if scenario == "ignorant":
        f_enrolls = os.path.join(eval_folder, "data", "eval_enrolls.txt")
    else:
        f_enrolls = os.path.join(eval_folder, "data", "anon_eval_enrolls.txt")


    # Optional: Set a seed for reproducibility
    if seed is not None:
        random.seed(seed)

    new_df = f_trials
    # Collect all unique speaker_ids
    all_speaker_ids = set()
    with open(new_df, 'r') as file:
        for line in file:
            obj = json.loads(line.strip())
            all_speaker_ids.add(obj["speaker_id"])

    # Generate speakers_list based on the provided upper_bound, lower_bound, and increment
    speakers_list = list(range(start_speakers, end_speakers + 1, increment_speakers))

    for max_speakers in speakers_list:
        with open(write_file, "a") as f:
                    f.write(f"\n")
        for iteration in range(iterations_per_group):

            chosen_ids = random.sample(list(all_speaker_ids), min(max_speakers, len(all_speaker_ids)))

            trials_ids = []
            with open(f_trials, 'r') as trial_file:
                for line in trial_file:
                    obj = json.loads(line.strip())
                    if obj["speaker_id"] in chosen_ids:
                        trials_ids.append(obj)
            filtered_trials = f"filtered_eval_trial.txt"
            with open(filtered_trials, 'w') as temp_file:
                for entry in trials_ids:
                    temp_file.write(json.dumps(entry) + "\n")
                    
            enrolls_ids = []
            with open(f_enrolls, 'r') as enroll_file:
                for line in enroll_file:
                    obj = json.loads(line.strip())
                    if obj["speaker_id"]  in chosen_ids:
                        enrolls_ids.append(obj)
        
                filtered_enrolls = f"filtered_eval_enroll.txt"
                with open(filtered_enrolls, 'w') as temp_file:
                    for entry in enrolls_ids:
                        temp_file.write(json.dumps(entry) + "\n")
        
                # compute SpkId vectors of all utts and map them to PLDA space
                start = time.time()
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
                
                 # Compute EER
                fpr, tpr, thresholds, key = compute_eer(unique_pairs[:, 0], unique_pairs[:, 1], llr_avgs)
                eer = (fpr[key] + (1 - tpr[key])) / 2
                end = time.time()

                #ASV Config
                print(f"Iteration: {iteration}, Num_speakers: {max_speakers}, {thresholds[key]}, {eer}")
                # print("Same-speaker accuracy:", np.sum(llrs[pairs[:, 0] == pairs[:, 1]] > thresholds[key]) / len(pairs))
                # print("Different-speaker accuracy:", np.sum(llrs[pairs[:, 0] != pairs[:, 1]] < thresholds[key]) / len(pairs))
                print("Duration in seconds:", end - start)


                with open(write_file, "a") as f:
                    f.write(f"{seed} {args.file} {max_speakers} {len(unique_pairs)} {len(pairs)} {thresholds[key]} {eer} {end-start} \n")

def main(args):
    """
    1. Select the trial and enrollment folders, which contain anonymized LibriSpeech file.
    2. Perform filtering strategy of the evaluation dataset
    3. Perform privacy evaluation: compute of SpkId vectors, perform PLDA mapping and compute EER. 
    """

    start_speakers = args.start_speakers  # Lower Bound for number of speakers
    end_speakers = args.upper_bound  # Upper Bound for number of speakers
    increment_speakers = args.increment # Increment number of speakers for each step
    iterations_per_group = args.iter # Number of times the sampling process should be repeated for each group size
    seed = args.seed # Seed for random number generation for reproducibility
    write_file =args.write_file

    #Initialize file
    with open(write_file, "a") as f:
        f.write(f"seed file speakers unique_pairs pairs threshold eer comput_time\n")

    sample_and_filter_speakers(args, write_file, start_speakers, end_speakers, increment_speakers, iterations_per_group, seed)
    
    # Remove tempory files.
    # os.remove("filtered_eval_enroll.txt")
    # os.remove("filtered_eval_trial.txt")
    print(f"Temp files have been deleted.")

    print("Upper bound of speakers reached!")
    print("Experimentation is completed!")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to eval folder")
    parser.add_argument("--write_file", type=str, help="Path to eval file", default= "spkanon_eval/temp.txt")
    parser.add_argument("--scen", type=str, help="Which scenario", default ="lazy-informed")
    parser.add_argument("--seed", type=int, help="Seed index", default=52)
    parser.add_argument("--file", type=int, help="File", default="1")
    parser.add_argument("--increment", type=int, help="Increment by", default=5)
    parser.add_argument("--iter", type=int, help="Number of iterations", default=5)
    parser.add_argument("--start_speakers", type=int, help="Start speaker index", default=10)
    parser.add_argument("--upper_bound", type=int, help="End speaker index", default=40)
    
    args = parser.parse_args()
    main(args)