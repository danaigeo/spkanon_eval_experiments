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
import time
from argparse import ArgumentParser

from spkanon_eval.setup_module import setup
from spkanon_eval.datamodules.dataset import load_audio
from spkanon_eval.evaluation.asv.asv_utils import analyse_results, compute_eer, compute_llrs
from spkanon_eval.evaluation.asv.spkid_plda import ASV
from spkanon_eval.evaluation.analysis import get_characteristics

def load_utterances(filename):
    """
    Function to load utterances from the original eval file
    
    Parameters:
    filename: The path to the eval file containing the anonymized utterances
    Returns:
    utterances_by_speaker: dictionary where keys are speaker IDs and values are lists of utterances

    """
    with open(filename, 'r') as file:
        lines = file.readlines()
    utterances_by_speaker = {}
    for line in lines:
        data = json.loads(line.strip())
        speaker_id = data['speaker_id']
        utterance = data['path']
        if speaker_id not in utterances_by_speaker:
            utterances_by_speaker[speaker_id] = []
        utterances_by_speaker[speaker_id].append(data)
    return utterances_by_speaker

# Function to filter out speakers
def filter_speakers_with_few_utterances(scenario, seed, all_utterances, min_utterances_per_speaker, file_name, stage):
    """
    Filter out the speakers that don't have enough utterances for experimentation
    Parameters:
    all_utterances (dict): A dictionary where each key is a speaker_id and each value is a list of utterances.
    min_utterances_per_speaker (int): Minimum number of utterances allowed per speaker.

    Returns:
    filtered_file: Filtered evaluation file (filtered_eval.txt)
    """
    selected_speakers = [] 
    for speaker_id, utterances in all_utterances.items():
        # Check if the speaker has enough utterances
        if len(utterances) >= min_utterances_per_speaker:
            # Randomly sample the specified number of utterances
            selected_speakers.append(speaker_id)
            
    ids = []
    with open(file_name, 'r') as file:
        for line in file:
            obj = json.loads(line.strip())
            if obj["speaker_id"] in selected_speakers:
                ids.append(obj)
                    
    filtered_file = f"filtered_eval.txt"
    with open(filtered_file, 'w') as temp_file:
        for entry in ids:
            temp_file.write(json.dumps(entry) + "\n")

    return filtered_file, len(selected_speakers)

def select_random_utterances(all_utterances, num_utterances_per_speaker):
    """
    Selects a random set of utterances for each speaker from a dictionary of utterances,
    excluding any speaker with fewer utterances than the specified number per speaker.

    Parameters:
    all_utterances (dict): A dictionary where each key is a speaker_id and each value is a list of utterances.
    num_utterances_per_speaker (int): Number of random utterances to select per speaker.

    Returns:
    list: A list containing the selected random utterances.
    """
    selected_utterances = []
    for speaker_id, utterances in all_utterances.items():
        # Check if the speaker has enough utterances
        selected = random.sample(utterances, min(len(utterances), num_utterances_per_speaker))
        for utterance in selected:
            selected_utterances.append(utterance)
    return selected_utterances

def filter_out_enroll_utterances(num_enrolls, f_trials, original_enrolls, seed, scenario, stage):
    """
    Selects a random set of utterances for each speaker from a dictionary of utterances,
    excluding any speaker with fewer utterances than the specified number per speaker.

    Parameters:
    all_utterances (dict): A dictionary where each key is a speaker_id and each value is a list of utterances.
    num_utterances_per_speaker (int): Number of random utterances to select per trial speaker.

    Returns:
    list: A list containing the selected random trial utterances.
    """
    
    with open(f_trials, "r") as trials_file:
        used_utterances = set()
        used_speaker_ids = set()
        for line in trials_file:
            utterance = json.loads(line.strip())
            used_utterances.add(utterance["text"])
            used_speaker_ids.add(utterance["speaker_id"])


    # Initialize a dictionary to store utterances per speaker
    utterances_per_speaker = {}
    
    # Read the original enrollment file, filter out speaker_ids that were excluded from the trials
    with open(original_enrolls, "r") as enrolls_file:
        for line in enrolls_file:
            utterance = json.loads(line.strip())
            speaker_id = utterance["speaker_id"]
            if speaker_id in used_speaker_ids:
                if speaker_id not in utterances_per_speaker:
                    utterances_per_speaker[speaker_id] = []
                utterances_per_speaker[speaker_id].append(utterance)
                
    # Select num_enrolls utterances per speaker, ensuring they are not in used_utterances
    selected_utterances = []
    for speaker_id, utterances in utterances_per_speaker.items():
        available_utterances = [utt for utt in utterances if utt["text"] not in used_utterances]
        if len(available_utterances) >= num_enrolls:
            selected_utterances.extend(random.sample(available_utterances, num_enrolls))
        else:
            selected_utterances.extend(random.sample(utterances, min(num_enrolls, len(utterances))))

    # Write selected enrollment utterances to a new file
    f_enrolls = f"filtered_eval_enrolls.txt"
    with open(f_enrolls, "w") as output_file:
        for utterance in selected_utterances:
            output_file.write(json.dumps(utterance) + "\n")
    return f_enrolls

def sample_and_filter_speakers(args, write_file, upper_bound, filtered_utterances_by_speaker, increment_trial, iterations_per_group, original_enrolls, num_speakers, root_folder, seed = None):
    """
    Function to perform ASV evaluation on the filtered eval files

    1. Loads model
    2. Resamples multiple times for each subset
    3. Performs ASV evaluation on filtered trial and enrollment evaluation files
    4. Computes EER
    5. Dumps results in the write_file

    Parameters:
    write_file: Path to file where results are dumped
    upper_bound: Upper bound for incrementation
    filtered_utterances_by_speaker: The filtered evaluation file with speaker IDs with enough utterances
    increment_trial: Number to increment the trials per speaker
    iterations_per_group: Defines the amount of resampling per group
    original_enrolls: Path to original enroll file (either anonymized or non-anonymized)
    num_speakers: Amount of filtered speakers
    """
    eval_folder = args.config
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

    num_utterances_per_speaker = args.start_trials

    # Optional: Set a seed for reproducibility
    if seed is not None:
        random.seed(seed)
        
    for increase in range(1, upper_bound):
        with open(write_file, "a") as f:
                f.write(f"\n")

        num_utterances_per_speaker = num_utterances_per_speaker + increment_trial
        for iteration in range(iterations_per_group):

            selected_utterances = select_random_utterances(filtered_utterances_by_speaker, num_utterances_per_speaker)
            # Save the selected trial utterances to a new file for each resample
            f_trials = f"filtered_eval_trials.txt"
            with open(f_trials, 'w') as file:
                if root_folder is not None:
                    for trial_utt in selected_utterances:
                        trial_utt["path"] = trial_utt["path"].replace(root_folder, trial_folder)
                        file.write(json.dumps(trial_utt) + '\n')

            # Fitler out num_enroll random enroll utterances that are not included trials
            f_enrolls = filter_out_enroll_utterances(args.num_enrolls, f_trials, original_enrolls, seed, scenario, "trial")

            # compute SpkId vectors of all utts and map them to PLDA space
            start = time.time()
            vecs, labels = dict(), dict()
            for name, f in zip(["trials", "enrolls"], [f_trials, f_enrolls]):
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
            print("No. of trials speakers:", len(labels["trials"]))
            print("No. of enroll speakers:", len(labels["enrolls"]))

            
            print("Averaging LLRs across speakers")
            print(f"No. of speaker pairs: {pairs.shape[0]}")
            
            # avg. LLRs across speakers
            unique_pairs, inverse = np.unique(pairs, axis=0, return_inverse=True)
            llr_avgs = np.bincount(inverse, weights=llrs) / np.bincount(inverse)
            print("There are", len(unique_pairs), "unique pairs of trial-enroll embeddings")

            trials = unique_pairs[:, 0]
            enrolls =  unique_pairs[:, 1]  

            # Compute EER
            fpr, tpr, thresholds, key = compute_eer(unique_pairs[:, 0], unique_pairs[:, 1], llr_avgs)
            eer = (fpr[key] + (1 - tpr[key])) / 2
            end = time.time()

            #ASV Config
            print(f"Iteration: {iteration}, Num_trials: {num_utterances_per_speaker}, {thresholds[key]}, {eer}")
            # print("Same-speaker accuracy:", np.sum(llrs[pairs[:, 0] == pairs[:, 1]] > thresholds[key]) / len(pairs))
            # print("Different-speaker accuracy:", np.sum(llrs[pairs[:, 0] != pairs[:, 1]] < thresholds[key]) / len(pairs))
            print("Duration in seconds:", end - start)

            with open(write_file, "a") as f:
                f.write(f"{seed} {file} {num_speakers} {num_utterances_per_speaker} {args.num_enrolls} {len(unique_pairs)} {len(pairs)} {thresholds[key]} {eer} {end-start} \n")



def main(args):
    """
    1. Select the trial and enrollment folders, which contain anonymized LibriSpeech file.
    2. Perform filtering strategy of the evaluation dataset
    3. Perform privacy evaluation: compute of SpkId vectors, perform PLDA mapping and compute EER. 
    """
    eval_folder = args.config # Selected log file for evaluation
    scenario = args.scen # Choose attack model for evaluation
    seed = args.seed # Seed for reproducibility
    min_utter = args.min_utter # Minimum number of overall utterances per speaker
    write_file =args.write_file # File to save the results
    iterations_per_group = args.iter # Number of times the sampling process should be repeated for each group size
    increment_trial = args.increment # Increment number of enrollment utterances for each step
    root_folder = "/ds/audio" # Root folder

    #Initialize file
    with open(write_file, "a") as f:
        f.write(f"seed file speakers trials enrolls unique_pairs pairs threshold eer comput_time\n")

 
    f_eval = os.path.join(eval_folder, "data", "eval.txt")
    # If the scenario is "lazy-informed", use th anonymized enrollment data
    if scenario == "ignorant":
        f_enrolls = os.path.join(eval_folder, "data", "eval_enrolls.txt")
    else:
        f_enrolls = os.path.join(eval_folder, "data", "anon_eval_enrolls.txt")

    utterances_by_speaker = load_utterances(f_eval) # Save the utterances per speaker in a dictionary

    # Filter out the speakers that have enough utterances to experiment with
    filtered_eval, num_speakers = filter_speakers_with_few_utterances(scenario, seed, utterances_by_speaker, min_utter, f_eval, "eval")
    print(f"No. of speakers: {num_speakers}, No. of enrolls: {args.num_enrolls}")

    filtered_utterances_by_speaker = load_utterances(filtered_eval)

    # Perform privacy evaluation
    sample_and_filter_speakers(args, write_file, args.end_trials, filtered_utterances_by_speaker, increment_trial, iterations_per_group, f_enrolls, num_speakers, root_folder, seed)

    # Remove tempory files.
    os.remove("filtered_eval.txt")
    os.remove("filtered_eval_enrolls.txt")
    os.remove("filtered_eval_trials.txt")
    print(f"Temp files have been deleted.")

    print("Upper bound of trials reached!")
    print("Experimentation is completed!")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to eval folder")
    parser.add_argument("--write_file", type=str, help="Path to write file", default= "spkanon_eval/temp.txt")
    parser.add_argument("--scen", type=str, help="Which scenario", default ="lazy-informed")
    parser.add_argument("--seed", type=int, help="Seed index", default=100)
    parser.add_argument("--file", type=int, help="File", default="1")
    parser.add_argument("--increment", type=int, help="Increment by", default=5)
    parser.add_argument("--iter", type=int, help="Number of iterations", default=5)
    parser.add_argument("--start_trials", type=int, help="Start: #Trial utterances per Speaker", default=5)
    parser.add_argument("--end_trials", type=int, help="Upper bound: #Trial utterances per Speaker", default=5)
    parser.add_argument("--num_enrolls", type=int, help="#Enrollment utterances per Speaker", default=20)
    parser.add_argument("--min_utter", type=int, help="Minimum utterances per Speaker", default=180)
    
    args = parser.parse_args()
    main(args)