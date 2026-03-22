import os
import torch
import torchaudio
import numpy as np
import random
import json
import laion_clap
import torch.nn.functional as F

from audioldm2 import build_model, text_to_audio, save_wave

# -------------------------
# CONFIG
# -------------------------
# BASE_PROMPT = "A music track played by an orchestra"
# BASE_PROMPT = "A live recording of a band"
BASE_PROMPT = "A music track"
CONCEPT_PROMPT_TEMPLATE = "A {} music track"
CONCEPTS = ["jazz", "classical", "hip hop", "EDM"]  # concepts we want to find seeds for, can be any words or phrases
CANDIDATE_SEEDS = list(range(0, 200))   # test 200 seeds
MODEL_NAME = "audioldm_48k"  # or "audioldm2-16k"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEEDS_PER_CONCEPT = 22
BEST_DIFF_THRESHOLD = 0.05  # only consider a seed valid if the similarity improvement is at least this much better than the base prompt
SAVE_AUDIO_PATH = "./seed_sampling_genre"  # set to None if you don't want to save generated audio

completed_concepts = set()  # to keep track of concepts we've already found valid seeds for, so we can stop early if desired

# -------------------------
# Determinism
# -------------------------
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

os.makedirs(SAVE_AUDIO_PATH, exist_ok=True)

sample_rate = 48000

# -------------------------
# Load AudioLDM2
# -------------------------
print("Loading AudioLDM2...")
audioldm = build_model(model_name=MODEL_NAME, device=DEVICE)

# -------------------------
# Load CLAP
# -------------------------
print("Loading CLAP...")
clap = laion_clap.CLAP_Module(enable_fusion=False)
clap.load_ckpt()
clap = clap.to(DEVICE)
clap.eval()

# Precompute text embeddings
text_embeddings = {}
for concept in CONCEPTS:
    text_np = clap.get_text_embedding([concept])
    text_embeddings[concept] = torch.from_numpy(text_np).to(DEVICE)

# -------------------------
# Function: Generate + Score
# -------------------------
def generate_and_score(seed, seed_output_path=None):
    set_seed(seed)

    print(f"\nGenerating audio for seed {seed} with base prompt...")

    waveform_base_raw = text_to_audio(
        audioldm,
        BASE_PROMPT,
        seed=seed,
        duration=5,
        guidance_scale=3.5,
        ddim_steps=200,
        batchsize=1,
        latent_t_per_second=25.6
    )

    if seed_output_path is not None:
        save_wave(waveform_base_raw, seed_output_path, name=BASE_PROMPT, samplerate=sample_rate)

    waveform_base = torch.from_numpy(waveform_base_raw).float()
    waveform_base = waveform_base.squeeze(1)   # remove channel dim

    # ensure correct sample rate
    # if "48k" not in MODEL_NAME:
    #     waveform_base = torchaudio.functional.resample(
    #         waveform_base,
    #         orig_freq=16000,
    #         new_freq=48000
    #     )

    waveform_base = waveform_base.to(DEVICE)

    with torch.no_grad():
        audio_embed = clap.get_audio_embedding_from_data(
            x=waveform_base,
            use_tensor=True
        ).to(DEVICE)

    base_similarities = {}
    for concept in CONCEPTS:
        sim = F.cosine_similarity(audio_embed, text_embeddings[concept])
        base_similarities[concept] = sim.item()

    for concept in CONCEPTS:

        if concept in completed_concepts:
            print(f"Already found valid seeds for concept '{concept}'. Skipping further testing for this concept.")
            continue
        concept_similarity = {}
        prompt = CONCEPT_PROMPT_TEMPLATE.format(concept)

        print(f"Generating audio for seed {seed} with concept prompt: '{prompt}'")

        waveform_concept_raw = text_to_audio(
            audioldm,
            prompt,
            seed=seed,
            duration=5,
            guidance_scale=3.5,
            ddim_steps=200,
            batchsize=1,
            latent_t_per_second=25.6
        )
        if seed_output_path is not None:
            save_wave(waveform_concept_raw, seed_output_path, name=prompt, samplerate=sample_rate)

        waveform_concept = torch.from_numpy(waveform_concept_raw).float()
        waveform_concept = waveform_concept.squeeze(1)   # remove channel dim

        # ensure correct sample rate
        # if "48k" not in MODEL_NAME:
        #     waveform_concept = torchaudio.functional.resample(
        #         waveform_concept,
        #         orig_freq=16000,
        #         new_freq=48000
        #     )

        waveform_concept = waveform_concept.to(DEVICE)

        with torch.no_grad():
            audio_embed_concept = clap.get_audio_embedding_from_data(
                x=waveform_concept,
                use_tensor=True
            ).to(DEVICE)
        
        for c in CONCEPTS:
            sim = F.cosine_similarity(audio_embed_concept, text_embeddings[c])
            concept_similarity[c] = sim.item()

        diffsimilarities = {c: concept_similarity[c] - base_similarities[c] for c in CONCEPTS}

        # get highest positive difference
        best_concept = max(diffsimilarities, key=diffsimilarities.get)
        best_diff = diffsimilarities[best_concept]

        # If the best_diff is positive, it means the model is more similar to the concept when we include it in the prompt, which is what we want
        if best_diff > BEST_DIFF_THRESHOLD and best_concept == concept:
            valid_seeds_per_concept[concept].append((seed, best_diff))
            print(f"Seed {seed} is VALID for concept '{concept}' with similarity improvement of {best_diff:.4f}")
            if len(valid_seeds_per_concept[concept]) >= SEEDS_PER_CONCEPT:
                completed_concepts.add(concept)
                print(f"Found {SEEDS_PER_CONCEPT} valid seeds for concept '{concept}'. Stopping search for this concept.")
        
        

    
# Main Filtering Loop
# -------------------------
valid_seeds_per_concept = {c: [] for c in CONCEPTS}

print("Starting seed filtering...")

for seed in CANDIDATE_SEEDS:
    if len(completed_concepts) == len(CONCEPTS):
        print("Found sufficient valid seeds for all concepts. Stopping search.")
        break
    # create a folder for this seed's audio outputs
    seed_output_path = os.path.join(SAVE_AUDIO_PATH, f"seed_{seed}")
    os.makedirs(seed_output_path, exist_ok=True)

    print(f"Testing seed {seed}")
    generate_and_score(seed, seed_output_path)
    # print a summary of how many valid seeds we have found so far for each concept
    print("Current valid seeds per concept:")
    for concept, seeds in valid_seeds_per_concept.items():
        print(f"  {concept}: {len(seeds)} valid seeds")


# -------------------------
# Save results
# -------------------------
with open("valid_seeds_genre.json", "w") as f:
    json.dump(valid_seeds_per_concept, f, indent=4)

print("\nValid seeds per concept:")
for concept, seeds in valid_seeds_per_concept.items():
    print(concept, ":", len(seeds))