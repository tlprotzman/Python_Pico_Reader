import pico_reader
import matplotlib.pyplot as plt
import awkward as ak
import numpy as np

pr = pico_reader.PicoDST()
pr.import_data("st_physics_21250020_raw_6500001.picoDst.root")

# Check EPD Data is correctly imported
# plt.hist(pr.epd_hits.nMip, 50)
# plt.show()

print(len(pr.p_t))

ring_sums = pr.epd_hits.generate_epd_hit_matrix()
print(ring_sums)