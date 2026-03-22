import torch
import torchaudio
import numpy as np
import random
import laion_clap
import torch.nn.functional as F
import json

#read json
file_path = "valid_seeds_instruments-band.json"
concept = "violin"
audio_path_temp = "seed_sampling_band_inst/seed_{}/A live recording of a band.wav"

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

count_negative = 0
for seed, _ in data[concept]:
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
        text_embed_np = model.get_text_embedding([concept])
        text_embed = torch.from_numpy(text_embed_np).to(device)
    similarity = F.cosine_similarity(audio_embed, text_embed)
    print(f"Similarity to '{concept}' for seed {seed}: {similarity.item():.4f}")
    if similarity.item() < 0.06:  # arbitrary threshold for "negative" seeds
        count_negative += 1

print(f"Total seeds tested: {len(data[concept])}")
print(f"Number of seeds with similarity < 0.05: {count_negative}")