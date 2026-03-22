import torch
import torchaudio
import numpy as np
import random
import laion_clap
import torch.nn.functional as F

# -----------------------
# 1. Make everything deterministic
# -----------------------
# def set_seed(seed=0):
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     np.random.seed(seed)
#     random.seed(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False

# set_seed(0)

device = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------
# 2. Load CLAP model (pure inference)
# -----------------------
model = laion_clap.CLAP_Module(enable_fusion=False)
model.load_ckpt()  # loads default pretrained checkpoint
model = model.to(device)
model.eval()

# -----------------------
# 3. Load Audio
# -----------------------
audio1_path = "output/02_03_2026_09_22_04/A music track.wav"
audio2_path = "output/02_03_2026_09_38_42/A EDM music track.wav"

waveform1, sr = torchaudio.load(audio1_path)
waveform2, sr = torchaudio.load(audio2_path)

if waveform1.shape[0] > 1:
    waveform1 = waveform1.mean(dim=0, keepdim=True)

if waveform2.shape[0] > 1:
    waveform2 = waveform2.mean(dim=0, keepdim=True)

if sr != 48000:
    waveform1 = torchaudio.functional.resample(
        waveform1, orig_freq=sr, new_freq=48000
    )
    waveform2 = torchaudio.functional.resample(
        waveform2, orig_freq=sr, new_freq=48000
    )

audio_tensor1 = waveform1.to(device)  # shape [1, T]
audio_tensor2 = waveform2.to(device)  # shape [1, T]

# -----------------------
# 4. Get embeddings
# -----------------------

concepts = ["jazz", "rock", "classical", "hip hop", "EDM"]
base_similarities = []
modified_similarities = []

for concept in concepts:
    # print(f"Text embedding for '{concept}':", text_embed.cpu().numpy())
    with torch.no_grad():
        audio1_embed = model.get_audio_embedding_from_data(
            x=audio_tensor1,  # [1, T]
            use_tensor=True
        ).to(device)
        audio2_embed = model.get_audio_embedding_from_data(
            x=audio_tensor2,  # [1, T]
            use_tensor=True
        ).to(device)
        text_embed_np = model.get_text_embedding([concept])
        text_embed = torch.from_numpy(text_embed_np).to(device)

    # -----------------------
    # 5. Cosine similarity
    # -----------------------
    similarity1 = F.cosine_similarity(audio1_embed, text_embed)
    similarity2 = F.cosine_similarity(audio2_embed, text_embed)

    print("Similarity for base prompt for concept '{}': {}".format(concept, similarity1.item()))
    print("Similarity for modified prompt for concept '{}': {}".format(concept, similarity2.item()))
    base_similarities.append(similarity1.item())
    modified_similarities.append(similarity2.item())

# get difference in similarity for each concept
similarity_differences = [mod - base for mod, base in zip(modified_similarities, base_similarities)]
print("\nSimilarity differences (modified - base) for each concept:")
for concept, diff in zip(concepts, similarity_differences):
    print(f"{concept}: {diff}")
