import pico_reader
import matplotlib.pyplot as plt
import awkward as ak
import numpy as np

pr = pico_reader.PicoDST()
# pr.import_data("st_physics_21250020_raw_6500001.picoDst.root")
pr.import_data(r"E:\2019Picos\14p5GeV\Runs\20094051.root")

# Check EPD Data is correctly imported
# plt.hist(pr.epd_hits.nMip, 50)
# plt.show()

print(len(pr.p_t))

ring_sums = pr.epd_hits.generate_epd_hit_matrix()
# print(ring_sums)


bins = 100
fig, ax = plt.subplots(8, 4, constrained_layout=True)
for i in range(8):
    for j in range(4):
        index = 4 * i + j
        ax[i][j].hist(ring_sums[i], bins)
        ax[i][j].set_title("Ring " + str(index + 1))
plt.show()
