import pico_reader
import matplotlib.pyplot as plt
import awkward as ak
import numpy as np

pr = pico_reader.PicoDST()
# pr.import_data("st_physics_21250020_raw_6500001.picoDst.root")
pr.import_data("/data/rhic/epd_centrality/7p7GeV/249/st_physics_21249043_raw_3000001.picoDst.root")

# Check EPD Data is correctly imported
# plt.hist(pr.epd_hits.nMip, 50)
# plt.show()

print(len(pr.p_t))

ring_sums = pr.epd_hits.generate_epd_hit_matrix()
# print(ring_sums)


bins = 60
fig, ax = plt.subplots(4, 4)
for i in range(4):
    for j in range(4):
        index = 4 * i + j
        ax[i][j].hist(ring_sums[i], bins)
        ax[i][j].set_title("Ring " + str(index + 1))
plt.tight_layout()
plt.show()