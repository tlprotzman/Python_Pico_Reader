import pico_reader
import numpy as np
import awkward as ak

pr = pico_reader.PicoDST()
pr.import_data("st_physics_21250020_raw_6500001.picoDst.root")

def basic_filter_1cm(events, index):
    return events.v_r[index] < 1  # Only get events within 1 cm of center

def basic_filter_2cm(events, index):
    return events.v_r[index] < 2  # Only get events within 2 cm of center


print(max(pr.v_r))

cut_events_1cm = pico_reader.Event_Cuts(pr, basic_filter_1cm)
cut_events_2cm = pico_reader.Event_Cuts(pr, basic_filter_2cm)
print("From", pr.num_events, ",", cut_events_1cm.num_events, "events have v_r < 1 cm")
print("From", pr.num_events, ",", cut_events_2cm.num_events, "events have v_r < 2 cm")

print(cut_events_1cm.v_r)
print(len(cut_events_1cm.v_r))