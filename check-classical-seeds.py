import torch
import torchaudio
import numpy as np
import random
import laion_clap
import torch.nn.functional as F
import json

#read json
file_path = "valid_seeds_genre.json"
audio_path_temp = "seed_sampling_genre/seed_{}/A music track.wav"
# concepts = ["EDM", "hip hop", "jazz", "classical"]
target_concept = "classical"
update_file = "filtered_seeds_genre.json"
new_file = "final_seeds_genre.json"

# -----------------------
# 1. Make everything deterministic
# -----------------------
def set_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

device = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------
# 2. Load CLAP model (pure inference)
# -----------------------
model = laion_clap.CLAP_Module(enable_fusion=False)
model.load_ckpt()  # loads default pretrained checkpoint
model = model.to(device)
model.eval()


with open(file_path, "r") as f:
    data = json.load(f)

threshold = 0.05  # arbitrary threshold for "negative" seeds

with open(update_file, "r") as f:
    #read full dict from json
    original_seeds = json.load(f)
    selected_seeds = original_seeds.get(target_concept, [])

for seed in range(26, 49):
    print(f"Seed: {seed}")
    audio_path = audio_path_temp.format(seed)
    set_seed(seed)

    waveform, sr = torchaudio.load(audio_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    audio_tensor = waveform.to(device)  # shape [1, T]
    # Get CLAP embedding
    with torch.no_grad():
        audio_embed = model.get_audio_embedding_from_data(
            x=audio_tensor,  # [1, T]
            use_tensor=True
        ).to(device)
        text_embed_np = model.get_text_embedding([target_concept])
        text_embed = torch.from_numpy(text_embed_np).to(device)
    similarity = F.cosine_similarity(audio_embed, text_embed)
    print(f"Similarity to '{target_concept}' for seed {seed}: {similarity.item():.4f}")
    if similarity.item() < threshold:  # arbitrary threshold for "negative" seeds
        print(f"Seed {seed} is a 'negative' seed with similarity {similarity.item():.4f}")
        selected_seeds.append([seed, similarity.item()])


print(f"\nSelected {len(selected_seeds)} seeds with similarity < {threshold} to '{target_concept}': {selected_seeds}")

# Save selected seeds to a new JSON file
original_seeds[target_concept] = selected_seeds
with open(new_file, "w") as f:
    json.dump(original_seeds, f, indent=4)
