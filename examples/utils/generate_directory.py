# coding: utf-8

import os


def make_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print("Done!")
    else:
        print(f"{directory} exists!")


def generate_directory_tree(task_name):

    root_dir = os.path.join(
        '/home/yangqi/dl/cheminfo/spoc_20210904/spoc', 'benchmark', task_name)
    dir_list = ['features', 'log', 'output']

    for dir in dir_list:
        dir = os.path.join(root_dir, dir)
        make_dir(dir)


if __name__ == '__main__':

    task_list = ['qm7',]
    for task in task_list:
        print()
        print(f"Prepare to generate directory tree: {task}")
        print(f"{'-'*20}")
        generate_directory_tree(task)
        print(f"{'-'*20}")
