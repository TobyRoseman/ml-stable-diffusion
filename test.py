import coremltools as ct

import numpy as np


p = "/Volumes/DevData/workspace/coreml-stable-diffusion-v1-5/original/compiled/TextEncoder.mlmodelc"


x = np.array([[49406,  2242, 11798,  3941,   530,   518,  6267,   267,  1400,
               9977,   267,   949,  3027, 49407, 49407, 49407, 49407, 49407,
               49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
               49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
               49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
               49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
               49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
               49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
               49407, 49407, 49407, 49407, 49407]]).astype('float32')

m = ct.models.CompiledMLModel(p)

model_output = m.predict({'input_ids': x})
y = model_output["last_hidden_state"]

print(y[0][0])
