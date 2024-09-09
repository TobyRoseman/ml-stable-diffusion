import numpy as np

shape = (1, 4, 64, 64)


def save_string_as_numpy(str_data, filename):
     str_data = str_data.replace(' ', ', ')
     a = eval(str_data)
     a = np.array(a).astype('float16')
     a = a.reshape(shape)
     np.save("swift_latents/"+filename, a)


# Break file into lines
with open('./allSwiftLatents.txt', 'r') as f:
    d = f.read().strip()
lines = d.split('\n')

for i in range(int(len(lines) / 4)):
    (name, str_val, junk1, junk2) = lines[i*4:i*4+4]
    assert junk1 == '===================================='
    assert all([s in junk2 for s in ('mean', 'median', 'last', 'step/sec')])

    save_string_as_numpy(str_val, name)
