import os

paths = ['clic', 'clic_no_qed', 'clic_no_qed_no_maxwell', 'idealized', 'idealized_no_qed', 'idealized_no_qed_no_maxwell', 'ilc', 'ilc_no_qed', 'ilc_no_qed_no_maxwell']

prefix = '/pscratch/sd/l/leosyam/'
suffix = '/particles_in/'

for p in paths:
    print(f'starting saving data for {p}')
    os.system(f'python3 id_track_mass.py {prefix + p + suffix} masssave_{p} fuck')
    print(f'finished saving data for {p}')
