import coremltools as ct
import numpy as np
import torch
from python_coreml_stable_diffusion.torch2coreml import compute_psnr


decoder = ct.models.MLModel('debug/Stable_Diffusion_version_CompVis_stable-diffusion-v1-4_vae_decoder.mlpackage/')
m_t = torch.jit.load('debug/decoder.pt')

for z_shape in ((1, 4, 64, 64), (1, 4, 32, 32)):
    z = np.random.rand(*z_shape).astype('float32')
    y_cm = decoder.predict({'z': z})['image']
    y_t = m_t(torch.tensor(z))[0].numpy()
    print(z_shape, compute_psnr(y_t, y_cm))

