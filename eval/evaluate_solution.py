import copy
import numpy as np
import os

from Evaluator.Evaluator import Evaluator
from options import MCSSOptions
from DataRW.S3DRW import S3DRW
from DataRW.wrong_annotatios import wrong_s3d_annotations_list

options = MCSSOptions()
opts = options.parse()
results_npy_dir = opts.results_dir

if __name__ == '__main__':
    if opts.scene_id == "val":
        opts.scene_id = "scene_03250" # Temp. value

        data_rw = S3DRW(opts)
        scene_list = data_rw.loader.scenes_list
        
        quant_result_dict = None
        quant_result_maskrcnn_dict = None
        scene_counter = 0
        for scene_ind, scene in enumerate(scene_list):
            if int(scene[6:]) in wrong_s3d_annotations_list:
                continue

            print("------------")
            curr_opts = copy.deepcopy(opts)
            curr_opts.scene_id = scene

            curr_data_rw = S3DRW(curr_opts)
            print("Running Evaluation for scene %s" % scene)

            evaluator = Evaluator(curr_data_rw, curr_opts)

            npy_path = os.path.join(results_npy_dir, scene[6:] + '.npy')

            room_polys = [p.astype(np.int64) for p in np.load(npy_path, allow_pickle=True)]
            room_polys = [poly for poly in room_polys if len(poly) > 0]

            quant_result_dict_scene = evaluator.evaluate_scene(room_polys=room_polys)

            if quant_result_dict is None:
                quant_result_dict = quant_result_dict_scene
            else:
                for k in quant_result_dict.keys():
                    quant_result_dict[k] += quant_result_dict_scene[k]

            scene_counter += 1

        for k in quant_result_dict.keys():
            quant_result_dict[k] /= float(scene_counter)

        print("Our: ", quant_result_dict)

        print("Ours")
        evaluator.print_res_str_for_latex(quant_result_dict)
