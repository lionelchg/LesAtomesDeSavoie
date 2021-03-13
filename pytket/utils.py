import os

fig_dir = 'figures/'

def create_dir(dir_name:str) -> None:
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
        