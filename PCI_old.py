from audioldm2 import text_to_audio, build_model, save_wave
import torch
import os

model_name = "audioldm_48k" # "audioldm2-music-665k" # "audioldm2-full"
device = "cuda" if torch.cuda.is_available() else "cpu"

concept = "violin"
base_text = "A live recording of a band"
PCI_text = "A live recording of a band with a {} solo".format(concept)
tau = 506 # at which diffusion step to apply PCI conditioning, added by me

seed = 42
duration = 10 # in seconds
guidance_scale = 3.5
ddim_steps = 200
n_candidate_gen_per_text = 1 # for generating multiple audio samples for the same text input
sample_rate = 48000
latent_t_per_second=12.8

save_path = "./PCI_audio"

os.makedirs(save_path, exist_ok=True)
audioldm2 = build_model(model_name=model_name, device=device)

waveform = text_to_audio(
            audioldm2,
            base_text,
            PCI_text=PCI_text, # added by me
            tau=tau, # added by me
            transcription="", # To avoid the model to ignore the last vocab
            seed=seed,
            duration=duration,
            guidance_scale=guidance_scale,
            ddim_steps=ddim_steps,
            n_candidate_gen_per_text=n_candidate_gen_per_text,
            batchsize=1,
            latent_t_per_second=latent_t_per_second
        )

file_name = f"PCI_{concept}_seed{seed}"

save_wave(waveform, save_path, name=file_name, samplerate=sample_rate)