from __future__ import absolute_import, division, print_function

import os
import argparse

file_dir = os.path.dirname(__file__)  # the directory that options.py resides in

class MCSSOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="MCSSFloor options")

        # PATHS
        self.parser.add_argument("--mcts_path",
                                 type=str,
                                 help="the name of the MonteFloorNet model",
                                 default="/media/sinisa/Sinisa_hdd_data/Sinisa_Projects/corridor_localisation/experiments/MonteFloorNet_experiments/room_shape_experiments/Structured3D_test/")


        self.parser.add_argument("--dataset_path",
                                 type=str,
                                 help="the name of the MonteFloorNet model",
                                 default="")

        self.parser.add_argument("--dataset_type",
                                 type=str,
                                 help="the name of the dataset type",
                                 choices=["floornet", "s3d", "fsp"])
        self.parser.add_argument("--scene_id",
                                 type=str,
                                 help="the name of the scene",
                                 default="0")

        self.parser.add_argument("--min_scene_ind",
                                 type=int,
                                 help="the name of the scene",
                                 default=0)
        self.parser.add_argument("--max_scene_ind",
                                 type=int,
                                 help="the name of the scene",
                                 default=251)
        self.parser.add_argument("--height",
                                 type=int,
                                 help="input image height",
                                 default=256)
        self.parser.add_argument("--width",
                                 type=int,
                                 help="input image width",
                                 default=256)
        self.parser.add_argument("--results_dir",
                                 type=str,
                                 help="path to the results dir that save the npy results",
                                 default="./outputs/s3d_polydiffuse_step10/npy")

    def parse(self):
        self.options = self.parser.parse_args()
        return self.options
