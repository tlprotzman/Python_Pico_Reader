import pico_reader

pr = pico_reader.PicoDST()
pr.import_data("st_physics_21250020_raw_6500001.picoDst.root")

print(pr.v_r)