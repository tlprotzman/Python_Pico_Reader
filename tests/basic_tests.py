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
        ax[i][j].hist(ring_sums[i], bins, histtype='step')
        ax[i][j].set_title("Ring " + str(index + 1))
plt.show()
plt.close()

plt.figure(figsize=(16, 9))
index = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
# Let's just sum the outer rings.
mip_sum = np.sum(ring_sums[index], axis=0)
plt.hist(mip_sum, bins, histtype='step', density=True)
plt.xlabel(r"$\Sigma_{nMIP}$", fontsize=15)
plt.ylabel(r"$\frac{dN}{d\Sigma_{nMIP}}$", fontsize=15)
plt.yscale('log')
plt.title("nMIP Sums for All Outer Rings", fontsize=20)
plt.tight_layout()
plt.show()

