from audioldm2 import text_to_audio, build_model, save_wave
import torch
import os
import numpy as np
import json

model_name = "audioldm_48k" # "audioldm2-music-665k" # "audioldm2-full"
device = "cuda" if torch.cuda.is_available() else "cpu"

concepts = ["EDM", "hip hop", "jazz", "classical"]
base_text = "A music track"
PCI_text_template = "A {} music track"
tau_interval = 50 # interval for tau values to apply PCI conditioning, set to a smaller number for more fine-grained control

duration = 10 # in seconds
guidance_scale = 3.5
ddim_steps = 200
n_candidate_gen_per_text = 1 # for generating multiple audio samples for the same text input
sample_rate = 48000
latent_t_per_second=12.8
PCI_sampling_n_seeds = 10 # number of seeds to sample for each concept for PCI generation, set to a smaller number for faster testing

seed_json = "final_seeds_genre.json"

base_dir = "./PCI_audio/{}".format(base_text.replace(" ", "_"))

def get_best_seeds(concept, seed_json, top_k=10):
    with open(seed_json, "r") as f:
        data = json.load(f)

    seeds = data[concept]
    # Sort seeds by least similarity score (assuming the second element in the tuple is the score)
    seeds.sort(key=lambda x: x[1], reverse=False)
    best_seeds = [seed for seed, score in seeds[:top_k]]
    return best_seeds

os.makedirs(base_dir, exist_ok=True)
audioldm2 = build_model(model_name=model_name, device=device)

tau_values = np.arange(1, 1000, tau_interval) # at which diffusion step to apply PCI conditioning, added by me
tau_values = tau_values[::-1]# reverse tau_values to apply PCI at later steps first

no_of_audios_to_generate = len(concepts) * PCI_sampling_n_seeds * len(tau_values)
print(f"Total number of audios to generate: {no_of_audios_to_generate}")

audios_generated = 0

for concept in concepts:
    # create a directory for each concept
    save_path = os.path.join(base_dir, concept)
    os.makedirs(save_path, exist_ok=True)

    best_seeds = get_best_seeds(concept, seed_json, top_k=PCI_sampling_n_seeds)

    for seed in best_seeds: # only use the best seeds for PCI generation, you can adjust this slicing as needed

        # create a directory for each seed
        seed_save_path = os.path.join(save_path, f"seed_{seed}")
        os.makedirs(seed_save_path, exist_ok=True)

        for tau in tau_values:
            
            print(f"Generating audio for concept '{concept}' with seed {seed} and PCI at step {tau}")

            waveform = text_to_audio(
                        audioldm2,
                        base_text,
                        PCI_text=PCI_text_template.format(concept), # added by me
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

            file_name = f"PCI_{concept}_seed{seed}_tau{tau}"

            save_wave(waveform, seed_save_path, name=file_name, samplerate=sample_rate)
            
            audios_generated += 1
            print(f"Generated audio {audios_generated}/{no_of_audios_to_generate}")