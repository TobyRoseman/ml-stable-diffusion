import coremltools as ct
import gc
import numpy as np
import pickle
import torch
from python_coreml_stable_diffusion.torch2coreml import compute_psnr


file_in = open('./input.pkl', 'rb')
x_pickle = pickle.load(file_in)
timestep = x_pickle[1]
encoder_hidden_states = x_pickle[2]
x_cm = {
    'timestep': timestep.numpy().astype('float16'),
    'encoder_hidden_states': encoder_hidden_states.numpy()
}

m_t = torch.jit.load('debug/unet.pt')
m_cm = ct.models.MLModel('debug/Stable_Diffusion_version_CompVis_stable-diffusion-v1-4_unet.mlpackage')


def check_match(sample):
    sample = sample.astype('float32')
    x_cm['sample'] = sample
    y_t = m_t(torch.tensor(sample), timestep, encoder_hidden_states)[0].numpy()
    y_cm = m_cm.predict(x_cm)['noise_pred']               
    print(sample.shape, compute_psnr(y_t, y_cm))


for sample in (np.random.rand(2, 4, 64, 64),
               np.random.rand(2, 4, 48, 48),
               np.random.rand(2, 4, 32, 32)):
    check_match(sample)
