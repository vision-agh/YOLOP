import numpy as np
import json

from .AutoDriveDataset import AutoDriveDataset
from .convert import convert, id_dict, id_dict_single
from tqdm import tqdm

single_cls = False       # just detect vehicle

class BddDataset(AutoDriveDataset):
    def __init__(self, cfg, is_train, inputsize, transform=None):
        super().__init__(cfg, is_train, inputsize, transform)
        self.db = self._get_db()
        self.cfg = cfg

    def _get_db(self):
        """
        get database from the annotation file

        Inputs:

        Returns:
        gt_db: (list)database   [a,b,c,...]
                a: (dictionary){'image':, 'information':, ......}
        image: image path
        mask: path of the segmetation label
        label: [cls_id, center_x//256, center_y//256, w//256, h//256] 256=IMAGE_SIZE
        """
        print('building database...')
        can_print = True
        gt_db = []
        height, width = self.shapes
        for mask in tqdm(list(self.mask_list)):
            mask_path = str(mask)
            image_path = mask_path.replace(str(self.mask_root), str(self.img_root)).replace(".png", ".jpg")
            lane_path= mask_path.replace(str(self.mask_root), str(self.lane_root))

        # for label in tqdm(list(self.label_list)):
        #     label_path = str(label)
        #     image_path = label_path.replace(str(self.label_root), str(self.img_root)).replace(".txt", ".png")
        #     lane_path = mask_path = image_path

            gt = np.zeros((0, 5))
            # num_lines = sum(1 for line in open(label_path))
            # gt = np.zeros((num_lines, 5))


            # cnt = 0
            # with open(label_path, 'r') as f:
            #     for line in f:
            #         data = line.split(" ")
            #         gt[cnt][0]= int(data[0])
            #         gt[cnt][1:] = data[1:]
            #         cnt+=1

            # print(mask_path)

            rec = [{
                'image': image_path,
                'label': gt,
                'mask': mask_path,
                'lane': lane_path
            }]

            gt_db += rec
        print('database build finish')
        return gt_db

    def filter_data(self, data):
        remain = []
        for obj in data:
            if 'box2d' in obj.keys():  # obj.has_key('box2d'):
                if single_cls:
                    if obj['category'] in id_dict_single.keys():
                        remain.append(obj)
                else:
                    remain.append(obj)
        return remain

    def evaluate(self, cfg, preds, output_dir, *args, **kwargs):
        """  
        """
        pass
