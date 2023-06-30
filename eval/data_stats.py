from S3DLoader.S3DLoader import S3DDataset
from itertools import chain


s3d_dataset_train = S3DDataset(None, './montefloor_data/train', 0, 3500, True, 'test')
s3d_dataset_test = S3DDataset(None, './montefloor_data/test', 0, 3500, True, 'test')
s3d_dataset_val = S3DDataset(None, './montefloor_data/val', 0, 3500, True, 'test')

all_num_rooms = []
all_num_corners = []

for sample in chain(s3d_dataset_train, s3d_dataset_val, s3d_dataset_val):
    polygons = sample['polygons_list']
    all_num_rooms.append(len(polygons))
    num_corners = [len(item) for item in polygons]
    all_num_corners.extend(num_corners)

