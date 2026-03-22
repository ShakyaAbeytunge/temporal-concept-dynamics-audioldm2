import json
import os

import os
import torch
import torchaudio
import numpy as np
import random
import laion_clap
import torch.nn.functional as F

# -------------------------
# Determinism
# -------------------------

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

# get all seeds used for a concept category from the folder names where {base_folder}/{subconcept}/seed_{seed}/{audio_file}.wav
def get_seeds_from_folder(folder_path):
    seeds = set()
    for item in os.listdir(folder_path):
        if item.startswith("seed_"):
            try:
                seed = int(item.split("_")[1])
                seeds.add(seed)
            except ValueError:
                print(f"Warning: Could not parse seed from folder name '{item}'")
                continue
        else:
            print(f"Warning: Folder name '{item}' does not start with 'seed_'")

    seeds = sorted(list(seeds))
    return seeds

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

concepts = ["classical", "jazz", "hip hop", "EDM"] # the concept categories we are interested in analyzing
base_audio_template = "seed_sampling_genre/seed_{}/A music track.wav"
concept_audio_template = "seed_sampling_genre/seed_{}/A {} music track.wav"
get_seeds_folder_template = "PCI_audio/A_music_track/{}"

thresholds_save_path = "PCI_analysis_plots/genre/thresholds.json"
os.makedirs(os.path.dirname(thresholds_save_path), exist_ok=True)

save_seeds_path = "PCI_analysis_plots/genre/seeds.json" # for later analysis, we save the seeds used for each concept category in a json file
os.makedirs(os.path.dirname(save_seeds_path), exist_ok=True)

print("Loading CLAP...")
clap = laion_clap.CLAP_Module(enable_fusion=False)
clap.load_ckpt()
clap = clap.to(DEVICE)
clap.eval()

similarity_diff = {} # to store the seed average similarity difference for each concept

seed_dict = {}
for concept in concepts:
    concept_folder = get_seeds_folder_template.format(concept)
    seeds = get_seeds_from_folder(concept_folder)
    seed_dict[concept] = seeds

    if concept == "classical":
        seeds = seeds[:5]

    for seed in seeds:
        set_seed(seed)

        concept_text_embed = clap.get_text_embedding([concept])
        concept_text_embed = F.normalize(torch.from_numpy(concept_text_embed).to(DEVICE), dim=-1)

        base_audio = torchaudio.load(base_audio_template.format(seed))[0].to(DEVICE)
        concept_audio = torchaudio.load(concept_audio_template.format(seed, concept))[0].to(DEVICE)

        with torch.no_grad():
            base_embed = clap.get_audio_embedding_from_data(
                    x=base_audio, use_tensor=True
                )
            concept_embed = clap.get_audio_embedding_from_data(
                    x=concept_audio, use_tensor=True
                )
        base_embed = F.normalize(base_embed, dim=-1)
        concept_embed = F.normalize(concept_embed, dim=-1)

        similarity_diff_value = (F.cosine_similarity(concept_embed, concept_text_embed) - F.cosine_similarity(base_embed, concept_text_embed)).item()

        if concept not in similarity_diff:
            similarity_diff[concept] = []
        similarity_diff[concept].append(similarity_diff_value)

# After processing all seeds, we can calculate the 25th, 50th, and 75th percentiles for each concept, and also for all concepts combined
thresholds = {}
all_diffs = []
for concept, diffs in similarity_diff.items():
    thresholds[concept] = {
        "25th_percentile": np.percentile(diffs, 25),
        "50th_percentile": np.percentile(diffs, 50),
        "75th_percentile": np.percentile(diffs, 75)
    }
    all_diffs.extend(diffs)
thresholds["overall"] = {
    "25th_percentile": np.percentile(all_diffs, 25),    
    "50th_percentile": np.percentile(all_diffs, 50),
    "75th_percentile": np.percentile(all_diffs, 75)
}

with open(save_seeds_path, "w") as f:
    json.dump(seed_dict, f, indent=4)

with open(thresholds_save_path, "w") as f:
    json.dump(thresholds, f, indent=4)