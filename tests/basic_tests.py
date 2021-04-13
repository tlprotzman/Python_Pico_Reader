import pico_reader
import matplotlib.pyplot as plt
import awkward as ak
import numpy as np

pr = pico_reader.PicoDST()
pr.import_data("st_physics_21250020_raw_6500001.picoDst.root")

# Check EPD Data is correctly imported
# plt.hist(pr.epd_hits.nMip, 50)
# plt.show()
print("mnMip Average:", np.average(pr.epd_hits.nMip))
print("mnMip std:", np.std(pr.epd_hits.nMip))