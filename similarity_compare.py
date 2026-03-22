import matplotlib.pyplot as plt
import json

import os
import torch
import torchaudio
import numpy as np
import random
import laion_clap
import torch.nn.functional as F

import seaborn as sns

sns.set_theme(style="whitegrid", context="paper")  
# context="talk" → good for presentations
# use "paper" if this is for a paper

plt.rcParams["grid.alpha"] = 0.6      # lower opacity (default ~0.8)
plt.rcParams["grid.linewidth"] = 0.6  # thinner lines
plt.rcParams["grid.color"] = "#cccccc"

plt.rcParams["figure.figsize"] = (8, 5)
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

# -------------------------
# Determinism
# -------------------------
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def get_seeds_from_folder(folder_path):
    #take seeds from the folder name, where the folder name is in the format "seed_{seed_num}" inside the given folder path
    seeds = []
    for item in os.listdir(folder_path):
        if item.startswith("seed_"):
            try:
                seed_num = int(item.split("_")[1])
                seeds.append(seed_num)
            except ValueError:
                continue
    return seeds

def moving_average(y, k=3):
    return np.convolve(y, np.ones(k)/k, mode='same')

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

explore_folder = "PCI_audio/A_live_recording_of_a_band" # change this to the folder containing the generated audios
concepts = ["violin", "trumpet", "flute"] # the concept categories we are interested in analyzing
base_audio_template = "seed_sampling_band_inst/seed_{}/A live recording of a band.wav" 
plots_save_path = "PCI_analysis_plots/inst_band" # folder to save the analysis plots
os.makedirs(plots_save_path, exist_ok=True)

# two curves for each concept: 
# 1) similarity of the PCI-generated audio to the concept averaged across seeds, for each time step (tau value), filled with standard deviation error bars to show variability across seeds
# 2) similarity difference between the PCI-generated audio and the base prompt audio to the concept, averaged across seeds, for each time step (tau value), also with standard deviation error bars

similarity_curves = {}

print("Loading CLAP...")
clap = laion_clap.CLAP_Module(enable_fusion=False)
clap.load_ckpt()
clap = clap.to(DEVICE)
clap.eval()

tau_values = list(range(1, 1000, 50)) # adjust this range and interval based on how many PCI-generated audios you have and at which time steps they were generated

# Precompute text embeddings for concepts
# concept_text_embeds = {}
# for concept in concepts:
#     text_np = clap.get_text_embedding([concept])
#     concept_text_embeds[concept] = F.normalize(torch.from_numpy(text_np).to(DEVICE), dim=-1)

for concept in concepts:
    
    print(f"\nProcessing concept: {concept}")
    
    concept_folder = os.path.join(explore_folder, concept)
    seeds = get_seeds_from_folder(concept_folder)

    mean_sim = []
    std_sim = []

    for tau in tau_values:
        sim_values = []

        for seed in seeds:
            # print(f"  Tau {tau}, Seed {seed}...")
            set_seed(seed)
            
            # PCI audio path
            pci_audio_path = os.path.join(
                concept_folder,
                f"seed_{seed}",
                f"PCI_{concept}_seed{seed}_tau{tau}.wav"
            )

            # Base audio path
            base_audio_path = base_audio_template.format(seed)

            if not os.path.exists(pci_audio_path) or not os.path.exists(base_audio_path):
                continue

            # Load PCI audio
            pci_waveform, sr = torchaudio.load(pci_audio_path)
            pci_waveform = pci_waveform.to(DEVICE)

            # Load Base audio
            base_waveform, sr = torchaudio.load(base_audio_path)
            base_waveform = base_waveform.to(DEVICE)

            # Get embeddings
            with torch.no_grad():
                pci_audio_embed = clap.get_audio_embedding_from_data(
                    x=pci_waveform, use_tensor=True
                )
                base_audio_embed = clap.get_audio_embedding_from_data(
                    x=base_waveform, use_tensor=True
                )

            pci_audio_embed = F.normalize(pci_audio_embed, dim=-1)
            base_audio_embed = F.normalize(base_audio_embed, dim=-1)

            # Cosine similarity between two audio embeddings
            sim_audio_embed = F.cosine_similarity(pci_audio_embed, base_audio_embed).item()
            sim_values.append(sim_audio_embed)

        if len(sim_values) > 0:
            # print(f"Tau {tau}: Mean similarity = {np.mean(sim_values):.4f}, Std similarity = {np.std(sim_values):.4f} across seeds")
            mean_sim.append(np.mean(sim_values))
            std_sim.append(np.std(sim_values))
        else:
            # print(f"Tau {tau}: No valid audio pairs found for similarity calculation.")
            mean_sim.append(0)
            std_sim.append(0)

    similarity_curves[concept] = (tau_values, mean_sim, std_sim)
    
# -------------------------
# Plotting
# -------------------------

# vertical lines for each concept
lines = {
    "violin": [751, 351],
    "trumpet": [351, 151],
    "flute": [601, 451]
}

# lines = {
#     "piano": [901, 501],
#     "drum": [951, 451]
# }

# lines = {
#     "classical": [551, 251],
#     "jazz": [801, 151],
#     "EDM": [951, 401],
#     "hip hop": [651, 251]
# }

# plot similarity curves, one plot per concept, reverse x-axis so that earlier time steps (lower tau) are on the right and later time steps (higher tau) are on the left
# std fill between (mean - std) and (mean + std) to show variability across seeds
for concept in concepts:
    tau_values, mean_sim, std_sim = similarity_curves[concept]
    plt.figure()
    plt.plot(tau_values, mean_sim)
    plt.fill_between(tau_values, np.array(mean_sim) - np.array(std_sim), np.array(mean_sim) + np.array(std_sim), alpha=0.3)
    plt.xlabel("Tau (time step of PCI conditioning)")
    plt.ylabel("Cosine Similarity")
    # add vertical lines for reference
    if concept in lines:
        # first line green dashed and shaded, second line red dashed and shaded
        line_tau = lines[concept][0]
        plt.axvline(x=line_tau, color='green', linestyle='--', label=f"Reference line at tau={line_tau}")
        plt.fill_betweenx([0, 1], 951, line_tau, color='green', alpha=0.1)
        line_tau = lines[concept][1]
        plt.axvline(x=line_tau, color='red', linestyle='--', label=f"Reference line at tau={line_tau}")
        plt.fill_betweenx([0, 1], line_tau, 1, color='red', alpha=0.1)

    # plt.title(f"Similarity of PCI-generated audio to '{concept}' across tau values")
    plt.gca().invert_xaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_save_path, f"{concept}_base_similarity_curve.png"))
    plt.close()

# average similarity curves across concepts
# avg_mean_sim = np.mean([similarity_curves[c][1] for c in concepts], axis=0)
# avg_std_sim = np.mean([similarity_curves[c][2] for c in concepts], axis=0)
# plt.figure()
# plt.plot(tau_values, avg_mean_sim, label="Average generation similarity for genre concepts")
# plt.fill_between(tau_values, avg_mean_sim - avg_std_sim, avg_mean_sim + avg_std_sim, alpha=0.3)
# plt.xlabel("Tau (time step of PCI conditioning)")
# plt.ylabel("Cosine Similarity")
# # plt.title("Average Similarity of PCI-generated audio to Concepts across tau values")
# plt.legend()
# plt.gca().invert_xaxis()
# plt.tight_layout()
# plt.savefig(os.path.join(plots_save_path, f"average_similarity_curve.png"))
# plt.close()