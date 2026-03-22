import json
import re
import torch
import os
import numpy as np
import matplotlib.pyplot as plt

# import seaborn as sns

# sns.set_theme(style="whitegrid", context="paper")  
# # context="talk" → good for presentations
# # use "paper" if this is for a paper

# plt.rcParams["grid.alpha"] = 0.6      # lower opacity (default ~0.8)
# plt.rcParams["grid.linewidth"] = 0.6  # thinner lines
# plt.rcParams["grid.color"] = "#cccccc"

# plt.rcParams["figure.figsize"] = (8, 5)
# plt.rcParams["axes.spines.top"] = False
# plt.rcParams["axes.spines.right"] = False

prompt_pairs = json.load(open("prompt_pairs.json", "r"))

def sanitize_text(text, max_len=80):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = text.strip().replace(" ", "_")
    return text[:max_len]

def load_prompt_generations(prompt_dir):
    """
    Loads all generations for a single prompt.
    Returns: Tensor [G, S, C, T, F]
    """
    gens = []
    timesteps = None

    for sample_name in sorted(os.listdir(prompt_dir)):
        sample_dir = os.path.join(prompt_dir, sample_name)
        if not sample_name.startswith("sample_"):
            continue

        # data = torch.load(os.path.join(sample_dir, "pred_x0.pt"))
        # pred_x0 = data["pred_x0"] # [S, C, T, F]

        data = torch.load(os.path.join(sample_dir, "x_inter.pt"))
        pred_x0 = data["x_inter"] # [S, C, T, F]

        gens.append(pred_x0)

        if timesteps is None:
            timesteps = data["t"]

    gens = torch.stack(gens)  # [G, S, C, T, F]
    return gens, timesteps

def latent_l2(a, b):
    """
    a, b: [G, S, C, T, F]
    Returns: [S] divergence per diffusion step
    """
    diff = a - b
    diff = diff.flatten(2)      # [G, S, D]
    diff = torch.norm(diff, dim=-1)  # [G, S]
    return diff.mean(dim=0)     # average over generations → [S]

results = {}  # {pair_id: {"t": timesteps, "divergence": divergence, "p1": prompt1, "p2": prompt2}}

base_log_dir = "logs/intermediates"

def compute_pairwise_correlation(curves):
    """
    curves: [N_pairs, S] array
    returns: mean correlation across all pairs
    """
    N = curves.shape[0]
    corrs = []
    for i in range(N):
        for j in range(i + 1, N):
            r = np.corrcoef(curves[i], curves[j])[0, 1]
            corrs.append(r)
    return np.array(corrs)

for category, pairs in prompt_pairs.items():
    print(f"Processing category: {category}")
    results[category] = {}
    
    for pair in pairs:
        print(f"  Pair: {pair['id']}")
        name = pair["id"]
        p1, p2 = pair["prompt_a"], pair["prompt_b"]

        dir1 = os.path.join(base_log_dir, sanitize_text(p1))
        dir2 = os.path.join(base_log_dir, sanitize_text(p2))

        z1, timesteps = load_prompt_generations(dir1)
        z2, _ = load_prompt_generations(dir2)

        divergence = latent_l2(z1, z2)  # [S]

        results[category][name] = {
            "t": timesteps,
            "divergence": divergence.cpu().numpy(),
            "p1": p1,
            "p2": p2,
        }
        
for category, pairs in results.items():
    # stack all divergences for this concept
    all_divs = []
    for pair_id, data in pairs.items():
        all_divs.append(data["divergence"])
    all_divs = np.stack(all_divs)  # [N_pairs, S]
    
    # compute pairwise correlation
    corrs = compute_pairwise_correlation(all_divs)
    print(f"{category} - Mean correlation between pairs: {corrs.mean():.3f} ± {corrs.std():.3f}")

for category, pairs in results.items():
    plt.figure(figsize=(10, 6))
    
    all_divs = []
    for pair_id, data in pairs.items():
        all_divs.append(data["divergence"])
    all_divs = np.stack(all_divs)  # [N_pairs, S]
    
    mean_curve = all_divs.mean(axis=0)
    std_curve = all_divs.std(axis=0)
    timesteps = data["t"]
    
    plt.plot(timesteps, mean_curve, label=f"{category} mean", color="blue")
    plt.fill_between(timesteps, mean_curve - std_curve, mean_curve + std_curve, color="blue", alpha=0.3)
    
    plt.title(f"Mean ± Std Divergence - {category}")
    plt.xlabel("Diffusion Step")
    plt.ylabel("Average L2 Divergence")
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"{category}_mean_std.png")
    plt.show()

    plt.figure(figsize=(10, 6))


# for category, pairs in results.items():
#     all_divs = np.stack([data["divergence"] for data in pairs.values()])
#     mean_curve = all_divs.mean(axis=0)
#     plt.plot(timesteps, mean_curve, label=category)
# plt.title("Concept-wise Mean Divergence")
# plt.xlabel("Diffusion Step")
# plt.ylabel("Average L2 Divergence")
# plt.grid()
# plt.legend()
# plt.tight_layout()
# plt.savefig("concept_mean_comparison.png")
# plt.show()

from itertools import combinations

concepts = list(results.keys())
within_corrs = []
between_corrs = []

# for category, pairs in results.items():
#     all_divs = np.stack([data["divergence"] for data in pairs.values()])
#     within_corrs.extend(compute_pairwise_correlation(all_divs))

# for c1, c2 in combinations(concepts, 2):
#     divs1 = np.stack([data["divergence"] for data in results[c1].values()])
#     divs2 = np.stack([data["divergence"] for data in results[c2].values()])
#     # pairwise correlation across concepts
#     for i in range(divs1.shape[0]):
#         for j in range(divs2.shape[0]):
#             r = np.corrcoef(divs1[i], divs2[j])[0, 1]
#             between_corrs.append(r)

# print(f"Within-concept mean correlation: {np.mean(within_corrs):.3f}")
# print(f"Between-concept mean correlation: {np.mean(between_corrs):.3f}")

# plot by category, 5 pairs per category, all in one plot

# for category, pairs in results.items():
#     plt.figure(figsize=(10, 6))
#     for pair_id, data in pairs.items():
#         t = data["t"]
#         divergence = data["divergence"]
#         plt.plot(t, divergence, label=f"{pair_id}")
    
#     plt.title(f"Divergence over Diffusion Steps - {category}")
#     plt.xlabel("Diffusion Step")
#     plt.ylabel("Average L2 Divergence")
#     plt.legend()
#     plt.grid()
#     plt.tight_layout()
#     plt.savefig(f"{category}_divergence.png")
#     plt.show()