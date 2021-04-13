import pico_reader
import numpy as np
import awkward as ak

pr = pico_reader.PicoDST()
pr.import_data("st_physics_21250020_raw_6500001.picoDst.root")

def basic_filter(events, index):
    return events.v_r[index] < 1000  # Only get events within 1 cm of center


print(max(pr.v_r))

cut_events = pico_reader.Event_Cuts(pr, basic_filter)
print("From", pr.num_events, ",", cut_events.num_events, "events remain")
