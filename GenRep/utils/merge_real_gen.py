from pathlib import Path
import shutil
import os


src_dir='../data/gen_train650'
dest_dir='../data/train'

for _dir in os.listdir(dest_dir):
    src_path=Path(src_dir).joinpath(_dir)
    dest_path=Path(dest_dir).joinpath(_dir)
    for src_file in src_path.glob('*'):
        shutil.copy(src_file,dest_path)
    print(f"{_dir} complete...")

for _dir in os.listdir(dest_dir):
    dest_path=Path(dest_dir).joinpath(_dir)
    num_file=len(os.listdir(dest_path))
    print(f"{_dir}: {num_file}")
