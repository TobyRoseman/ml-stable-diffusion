import numpy as np

import coremltools as ct


def compute_psnr(a, b):
    """
    Compute Peak Signal to Noise Ratio
    """
    assert len(a) == len(b)
    max_b = np.abs(b).max()
    sumdeltasq = 0.0

    sumdeltasq = ((a - b) * (a - b)).sum()       # XXX: np.square(a-b).sum()

    sumdeltasq /= b.size
    sumdeltasq = np.sqrt(sumdeltasq)        

    eps = 1e-5
    eps2 = 1e-10             # XXX: why two different eps?
    psnr = 20 * np.log10((max_b + eps) / (sumdeltasq + eps2))

    return psnr


'''
for i in range(4):
    bnns = np.load(f"./BNNS/latents-{i}.npy")
    cc = np.load(f"./classic_cpu/latents-{i}.npy")
    print(compute_psnr(bnns, cc))
'''


'''
for i in range(4):
    bnns = np.load(f"./BNNS/noise_pred-{i}.npy")
    cc = np.load(f"./classic_cpu/noise_pred-{i}.npy")
    print(compute_psnr(bnns, cc))

print("\n\n")

for i in range(4):
    bnns = np.load(f"./BNNS/combinded-noise_pred-{i}.npy")
    cc = np.load(f"./classic_cpu/combinded-noise_pred-{i}.npy")
    print(compute_psnr(bnns, cc))
'''

'''
bnns_np0 = np.load(f"./BNNS/noise_pred-0.npy")
cc_np0 = np.load(f"./classic_cpu/noise_pred-0.npy")

print(compute_psnr(bnns_np0, cc_np0))

guidance_scale = 7.5
def combine(noise_pred):
    noise_pred_uncond, noise_pred_text = np.split(noise_pred, 2)
    noise_pred = noise_pred_uncond + guidance_scale * (
        noise_pred_text - noise_pred_uncond)
    return noise_pred

print(compute_psnr(combine(bnns_np0), combine(cc_np0)))
'''


'''
# Combining the unconditioned and conditioned is what causes biggest PSNR diff

bnns_np0 = np.load(f"./BNNS/noise_pred-0.npy")
cc_np0 = np.load(f"./classic_cpu/noise_pred-0.npy")

print(compute_psnr(bnns_np0, cc_np0))
#61.075099488653635

bnns_noise_pred_uncond, bnns_noise_pred_text = np.split(bnns_np0, 2)
cc_noise_pred_uncond, cc_noise_pred_text = np.split(cc_np0, 2)


print(compute_psnr(bnns_noise_pred_uncond, cc_noise_pred_uncond))
# 60.64123010943742
print(compute_psnr(bnns_noise_pred_text, cc_noise_pred_text))
# 61.54892167676418


print(
    compute_psnr(
        bnns_noise_pred_text - bnns_noise_pred_uncond,
        cc_noise_pred_text - cc_noise_pred_uncond
    )
)
# 28.73738921393713


print(
    compute_psnr(
        bnns_noise_pred_uncond + guidance_scale * (bnns_noise_pred_text - bnns_noise_pred_uncond),
        cc_noise_pred_uncond + guidance_scale * (cc_noise_pred_text - cc_noise_pred_uncond)
    )
)
# 41.357129864339456
'''


'''
# init latents match

bnns_init_l = np.load('./BNNS/init-latents.npy')

cc_init_l = np.load('./classic_cpu//init-latents.npy')

compute_psnr(cc_init_l, bnns_init_l)
# 212.18423664406032
'''


'''
# Does batch order matter? -- doesn't seem to.
# [('sample', (2, 4, 64, 64)), ('timestep', (2,)), ('encoder_hidden_states', (2, 768, 1, 77))]

cm = ct.models.CompiledMLModel('../coreml-stable-diffusion-v1-5/original/compiled/Unet.mlmodelc')

text_embeddings1 = np.load("./BNNS/text_embeddings.npy")

latents = np.load("./BNNS/init-latents.npy")
latent_model_input = np.concatenate([latents] * 2)

t = 1

noise_pred1 = cm.predict(
    {
        'sample': latent_model_input.astype(np.float16),
        'timestep': np.array([t, t], np.float16),
        'encoder_hidden_states': text_embeddings1.astype(np.float16),
        #**unet_additional_kwargs,
    }
)["noise_pred"]
noise_pred_uncond1, noise_pred_text1 = np.split(noise_pred1, 2)


text_embeddings2 = np.flip(text_embeddings1, axis=0)
assert((text_embeddings2[0] == text_embeddings1[1]).all())
assert((text_embeddings2[1] == text_embeddings1[0]).all())

noise_pred2 = cm.predict(
    {
        'sample': latent_model_input.astype(np.float16),
        'timestep': np.array([t, t], np.float16),
        'encoder_hidden_states': text_embeddings2.astype(np.float16),
        #**unet_additional_kwargs,
    }
)["noise_pred"]
noise_pred_text2, noise_pred_uncond2 = np.split(noise_pred2, 2)

print(compute_psnr(noise_pred_uncond1, noise_pred_uncond2))
print(compute_psnr(noise_pred_text1, noise_pred_text2))

#76.27658316145036
#75.96402567262332
'''
