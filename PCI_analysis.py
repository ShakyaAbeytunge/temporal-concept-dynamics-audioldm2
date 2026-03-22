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

explore_folder = "PCI_audio/A_music_track" # change this to the folder containing the generated audios
concepts = ["classical", "jazz", "EDM", "hip hop"] # the concept categories we are interested in analyzing
base_audio_template = "seed_sampling_genre/seed_{}/A music track.wav" 
plots_save_path = "PCI_analysis_plots/genre" # folder to save the analysis plots
os.makedirs(plots_save_path, exist_ok=True)

# two curves for each concept: 
# 1) similarity of the PCI-generated audio to the concept averaged across seeds, for each time step (tau value), filled with standard deviation error bars to show variability across seeds
# 2) similarity difference between the PCI-generated audio and the base prompt audio to the concept, averaged across seeds, for each time step (tau value), also with standard deviation error bars

similarity_curves = {}
similarity_diff_curves = {}
success_curves = {}

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

threshold_path = "PCI_analysis_plots/genre/thresholds.json"
if os.path.exists(threshold_path):
    with open(threshold_path, "r") as f:
        thresholds = json.load(f)

panelty_rate = 0.25
threshold = thresholds["overall"]["50th_percentile"] * panelty_rate # set a global threshold for determining "success" of PCI based on the similarity difference curve analysis

for concept in concepts:
    
    print(f"\nProcessing concept: {concept}")
    
    concept_folder = os.path.join(explore_folder, concept)
    seeds = get_seeds_from_folder(concept_folder)

    mean_sim = []
    std_sim = []
    mean_diff = []
    std_diff = []
    success_curve = []

    for tau in tau_values:
        sim_values = []
        diff_values = []
        success_count = 0

        for seed in seeds:
            set_seed(seed)

            concept_text_embed = clap.get_text_embedding([concept])
            concept_text_embed = F.normalize(torch.from_numpy(concept_text_embed).to(DEVICE), dim=-1)
            
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

            # Cosine similarity
            sim_pci = (pci_audio_embed @ concept_text_embed.T).item()
            sim_base = (base_audio_embed @ concept_text_embed.T).item()

            sim_values.append(sim_pci)
            diff = sim_pci - sim_base
            diff_values.append(diff)

            # Success decision
            # get similarity against all concepts to see if the PCI audio is most similar to the target concept compared to the other concepts
            # other_concepts = [c for c in concepts if c != concept]
            # other_concept_embeds = torch.cat([concept_text_embeds[c] for c in other_concepts], dim=0) # shape [num_other_concepts, embed_dim]
            # sim_to_other_concepts = pci_audio_embed @ other_concept_embeds.T # shape [1, num_other_concepts]
            # max_sim_other = sim_to_other_concepts.max().item()
            # if sim_pci > max_sim_other and sim_pci > sim_base:
            #     success_count += 1
            if diff > threshold:
                success_count += 1

        if len(sim_values) > 0:
            mean_sim.append(np.mean(sim_values))
            std_sim.append(np.std(sim_values))
            mean_diff.append(np.mean(diff_values))
            std_diff.append(np.std(diff_values))
            success_curve.append(success_count / len(sim_values))
        else:
            mean_sim.append(0)
            std_sim.append(0)
            mean_diff.append(0)
            std_diff.append(0)
            success_curve.append(0)

    similarity_curves[concept] = (tau_values, mean_sim, std_sim)
    similarity_diff_curves[concept] = (tau_values, mean_diff, std_diff)
    success_curves[concept] = (tau_values, success_curve) # add final point at tau=1000 (end of generation) where we expect all to be successful since the concept is fully included in the prompt
# -------------------------
# Plotting
# -------------------------

# plot similarity curves, one plot per concept, reverse x-axis so that earlier time steps (lower tau) are on the right and later time steps (higher tau) are on the left
# std fill between (mean - std) and (mean + std) to show variability across seeds
for concept in concepts:
    tau_values, mean_sim, std_sim = similarity_curves[concept]
    plt.figure()
    plt.plot(tau_values, mean_sim, label=f"Generation similarity to text '{concept}'")
    plt.fill_between(tau_values, np.array(mean_sim) - np.array(std_sim), np.array(mean_sim) + np.array(std_sim), alpha=0.3)
    plt.xlabel("Tau (time step of PCI conditioning)")
    plt.ylabel("Cosine Similarity")
    # plt.title(f"Similarity of PCI-generated audio to '{concept}' across tau values")
    plt.legend()
    plt.gca().invert_xaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_save_path, f"{concept}_similarity_curve.png"))
    plt.close()

    tau_values, mean_diff, std_diff = similarity_diff_curves[concept]
    plt.figure()
    plt.plot(tau_values, mean_diff, label=f"Generation similarity to text '{concept}'")
    plt.fill_between(tau_values, np.array(mean_diff) - np.array(std_diff), np.array(mean_diff) + np.array(std_diff), alpha=0.3)
    plt.xlabel("Tau (time step of PCI conditioning)")
    plt.ylabel("Cosine Similarity Difference")
    # plt.title(f"Similarity Difference (PCI - Base) to '{concept}' across tau values")
    plt.legend()
    plt.gca().invert_xaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_save_path, f"{concept}_similarity_difference_curve.png"))
    plt.close()

    tau_values, success_rate = success_curves[concept]
    plt.figure()
    plt.plot(tau_values + [1000], success_rate + [1.0], label=f"Success Rate for concept '{concept}'")
    # --- Upper threshold (0.9) ---
    plt.axhline(0.8, linestyle="--", linewidth=1, color="green")

    plt.fill_between(
        tau_values + [1000],
        0.8,
        1.0,
        color="green",
        alpha=0.08
    )

    # --- Lower threshold (0.1) ---
    plt.axhline(0.2, linestyle="--", linewidth=1, color="red")

    plt.fill_between(
        tau_values + [1000],
        0.0,
        0.2,
        color="red",
        alpha=0.08
    )
    # plt.ylim(0, 1)
    plt.xlabel("Tau (time step of PCI conditioning)")
    plt.ylabel("Success Rate")
    # plt.title(f"Success Rate of PCI improving similarity to '{concept}' across tau values")
    plt.legend()
    plt.gca().invert_xaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_save_path, f"{concept}_success_curve.png"))
    plt.close()

# average similarity curves across concepts
avg_mean_sim = np.mean([similarity_curves[c][1] for c in concepts], axis=0)
avg_std_sim = np.mean([similarity_curves[c][2] for c in concepts], axis=0)
avg_mean_diff = np.mean([similarity_diff_curves[c][1] for c in concepts], axis=0)
avg_std_diff = np.mean([similarity_diff_curves[c][2] for c in concepts], axis=0)
# add initial success point at tau=1000 (end of generation) to 1
avg_success_rate = np.mean([success_curves[c][1] for c in concepts], axis=0)
plt.figure()
plt.plot(tau_values, avg_mean_sim, label="Average generation similarity for genre concepts")
plt.fill_between(tau_values, avg_mean_sim - avg_std_sim, avg_mean_sim + avg_std_sim, alpha=0.3)
plt.xlabel("Tau (time step of PCI conditioning)")
plt.ylabel("Cosine Similarity")
# plt.title("Average Similarity of PCI-generated audio to Concepts across tau values")
plt.legend()
plt.gca().invert_xaxis()
plt.tight_layout()
plt.savefig(os.path.join(plots_save_path, f"average_similarity_curve.png"))
plt.close()

plt.figure()
plt.plot(tau_values, avg_mean_diff, label="Average similarity difference (PCI - Base) for genre concepts")
plt.fill_between(tau_values, avg_mean_diff - avg_std_diff, avg_mean_diff + avg_std_diff, alpha=0.3)
plt.xlabel("Tau (time step of PCI conditioning)")
plt.ylabel("Cosine Similarity Difference")
# plt.title("Average Similarity Difference (PCI - Base) to Concepts across tau values")
plt.legend()
plt.gca().invert_xaxis()
plt.tight_layout()
plt.savefig(os.path.join(plots_save_path, f"average_similarity_difference_curve.png"))
plt.close()

smooth_success = moving_average(avg_success_rate.tolist() + [1.0], k=3)
smooth_success[-1] = 1.0 # for visualization purposes

plt.figure()
plt.plot(tau_values + [1000], smooth_success, label="Average Success Rate for genre concepts")
# plt.plot(tau_values, smooth_success, label="Smoothed Success Rate for Concepts")
# plt.ylim(0, 1)

# --- Upper threshold (0.9) ---
plt.axhline(0.8, linestyle="--", linewidth=1, color="green")

plt.fill_between(
    tau_values + [1000],
    0.8,
    1.0,
    color="green",
    alpha=0.08
)

# --- Lower threshold (0.1) ---
plt.axhline(0.2, linestyle="--", linewidth=1, color="red")

plt.fill_between(
    tau_values + [1000],
    0.0,
    0.2,
    color="red",
    alpha=0.08
)

plt.xlabel("Tau (time step of PCI conditioning)")
plt.ylabel("Success Rate")
# plt.title("Average Success Rate of PCI improving similarity to Concepts across tau values")
plt.legend()
plt.gca().invert_xaxis()
plt.tight_layout()
plt.savefig(os.path.join(plots_save_path, f"average_success_curve.png"))
plt.close()