import torch
import torch.utils.data as data

import os
import numpy as np
import numpy.random as npr
import scipy.io
import cv2
try:
    import cPickle  # Use cPickle on Python 2.7
except ImportError:
    import pickle as cPickle

from transforms3d.quaternions import mat2quat, quat2mat

import datasets
from fcn.config import cfg
from utils.blob import pad_im
from utils.se3 import *

class YCBVideo(data.Dataset, datasets.imdb):
    def __init__(self, image_set, ycb_video_path = None):

        self._name = 'ycb_video_' + image_set
        self._image_set = image_set
        self._ycb_video_path = self._get_default_path() if ycb_video_path is None \
                            else ycb_video_path

        path = os.path.join(self._ycb_video_path, 'data')
        if not os.path.exists(path):
            path = os.path.join(self._ycb_video_path, 'YCB_Video_Dataset/YCB_Video_Dataset/YCB_Video_Dataset/data')
        self._data_path = path

        self._model_path = os.path.join(datasets.ROOT_DIR, 'data', 'models')

        # define all the classes
        self._classes_all = ('__background__', 
        					 '002_master_chef_can',
        					 '003_cracker_box',
        					 '004_sugar_box',
        					 '005_tomato_soup_can',
        					 '006_mustard_bottle',
        					 '007_tuna_fish_can',
        					 '008_pudding_box',
        					 '009_gelatin_box',
        					 '010_potted_meat_can',
        					 '011_banana',
        					 '019_pitcher_base',
        					 '021_bleach_cleanser',
        					 '024_bowl',
        					 '025_mug',
        					 '035_power_drill',
        					 '036_wood_block',
        					 '037_scissors',
        					 '040_large_marker',
        					 '051_large_clamp',
        					 '052_extra_large_clamp',
        					 '061_foam_brick'
        					)
        self._num_classes_all = len(self._classes_all)
        self._class_colors_all = [(255, 255, 255), 
        						  (255, 0, 0),
        						  (0, 255, 0),
        						  (0, 0, 255),
        						  (255, 255, 0),
        						  (255, 0, 255),
        						  (0, 255, 255),
                              	  (128, 0, 0),
                              	  (0, 128, 0),
                              	  (0, 0, 128),
                              	  (128, 128, 0),
                              	  (128, 0, 128),
                              	  (0, 128, 128),
                              	  (64, 0, 0),
                              	  (0, 64, 0),
                              	  (0, 0, 64),
                              	  (64, 64, 0),
                              	  (64, 0, 64),
                              	  (0, 64, 64),
                              	  (192, 0, 0),
                              	  (0, 192, 0),
                              	  (0, 0, 192)]
        self._symmetry_all = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]).astype(np.float32)
        
        print("load object extents")        
        self._extents_all = self._load_object_extents()

        # print(self._extents_all)

        self._width = 640
        self._height = 480
        self._intrinsic_matrix = np.array([[1.066778e+03, 0.000000e+00, 3.129869e+02],
                                          [0.000000e+00, 1.067487e+03, 2.413109e+02],
                                          [0.000000e+00, 0.000000e+00, 1.000000e+00]])

        # select a subset of classes
        self._classes = [self._classes_all[i] for i in cfg.TRAIN.CLASSES]
        self._classes_test = [self._classes_all[i] for i in cfg.TEST.CLASSES]
        self._num_classes = len(self._classes)
        self._class_colors = [self._class_colors_all[i] for i in cfg.TRAIN.CLASSES]
        self._symmetry = self._symmetry_all[cfg.TRAIN.CLASSES]
        self._symmetry_test = self._symmetry_all[cfg.TEST.CLASSES]
        self._extents = self._extents_all[cfg.TRAIN.CLASSES]
        self._extents_test = self._extents_all[cfg.TEST.CLASSES]
        self._pixel_mean = cfg.PIXEL_MEANS / 255.0

        # train classes
        print("load object points")
        self._points, self._points_all, self._point_blob = \
            self._load_object_points(self._classes, self._extents, self._symmetry)

        # print(self._points[1].shape)
        # print(self._points_all[1, :10, :])
        # print(self._point_blob[1, :10, :])

        # test classes
        self._points_test, self._points_all_test, self._point_blob_test = \
            self._load_object_points(self._classes_test, self._extents_test, self._symmetry_test)

        self._class_to_ind = dict(zip(self._classes, range(self._num_classes)))
        print("load image set index")
        self._image_index = self._load_image_set_index(image_set)

        print("image index:", self._image_index)

        self._size = len(self._image_index)
        if self._size > cfg.TRAIN.MAX_ITERS_PER_EPOCH * cfg.TRAIN.IMS_PER_BATCH:
            self._size = cfg.TRAIN.MAX_ITERS_PER_EPOCH * cfg.TRAIN.IMS_PER_BATCH
        self._roidb = self.gt_roidb()

        print("roidb length:", len(self._roidb))

        assert os.path.exists(self._ycb_video_path), \
                'ycb_video path does not exist: {}'.format(self._ycb_video_path)
        assert os.path.exists(self._data_path), \
                'Data path does not exist: {}'.format(self._data_path)

    def __getitem__(self, index):

        is_syn = 0
        roidb = self._roidb[index]

        # Get the input image blob
        random_scale_ind = npr.randint(0, high=len(cfg.TRAIN.SCALES_BASE))
        im_blob, im_depth, im_scale, height, width = self._get_image_blob(roidb, random_scale_ind)

        print('image_color:', im_blob.shape)
        print('im_depth:', im_depth.shape)
        print('im_scale:', im_scale)

        # build the label blob
        label_blob, mask, meta_data_blob, pose_blob, gt_boxes, vertex_targets, vertex_weights \
            = self._get_label_blob(roidb, self._num_classes, im_scale, height, width)

        print('label:', label_blob.shape)
        print('mask:', mask.shape)
        print('meta_data:', meta_data_blob.shape)
        print('poses:', pose_blob.shape)
        print('gt_boxes:', gt_boxes.shape)

        print()

        is_syn = roidb['is_syn']
        im_info = np.array([im_blob.shape[1], im_blob.shape[2], im_scale, is_syn], dtype=np.float32)

        sample = {'image_color': im_blob,   # [3, 480, 640]
                  'im_depth': im_depth,     # [480, 640]
                  'label': label_blob,      # [num_classes, 480, 640]
                  'mask': mask,             # [3, 480, 640]
                  'meta_data': meta_data_blob, # [18, ]
                  'poses': pose_blob,       # [num_classes, 9]
                  'extents': self._extents,
                  'points': self._point_blob,
                  'symmetry': self._symmetry,
                  'gt_boxes': gt_boxes,     # [num_classes, 5]
                  'im_info': im_info,
                  'video_id': roidb['video_id'],
                  'image_id': roidb['image_id']}

        if cfg.TRAIN.VERTEX_REG:
            sample['vertex_targets'] = vertex_targets
            sample['vertex_weights'] = vertex_weights

        return sample

    def __len__(self):
        return self._size

    def _get_image_blob(self, roidb, scale_ind):

        # rgba
        rgba = pad_im(cv2.imread(roidb['image'], cv2.IMREAD_UNCHANGED), 16)
        if rgba.shape[2] == 4:
            im = np.copy(rgba[:,:,:3])
            alpha = rgba[:,:,3]
            I = np.where(alpha == 0)
            im[I[0], I[1], :] = 0
        else:
            im = rgba

        im_scale = cfg.TRAIN.SCALES_BASE[scale_ind]
        if im_scale != 1.0:
            im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
        height = im.shape[0]
        width = im.shape[1]

        if roidb['flipped']:
            im = im[:, ::-1, :]

        # chromatic transform
        if cfg.TRAIN.CHROMATIC and cfg.MODE == 'TRAIN' and np.random.rand(1) > 0.1:
            im = chromatic_transform(im)
        if cfg.TRAIN.ADD_NOISE and cfg.MODE == 'TRAIN' and np.random.rand(1) > 0.1:
            im = add_noise(im)
        im_tensor = torch.from_numpy(im) / 255.0
        im_tensor -= self._pixel_mean
        image_blob = im_tensor.permute(2, 0, 1).float()

        # depth image
        im_depth = pad_im(cv2.imread(roidb['depth'], cv2.IMREAD_UNCHANGED), 16)
        if im_scale != 1.0:
            im_depth = cv2.resize(im_depth, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_NEAREST)
        im_depth = im_depth.astype('float') / 10000.0

        return image_blob, im_depth, im_scale, height, width

    def _get_label_blob(self, roidb, num_classes, im_scale, height, width):
        """ build the label blob """

        meta_data = scipy.io.loadmat(roidb['meta_data'])
        meta_data['cls_indexes'] = meta_data['cls_indexes'].flatten()
        classes = np.array(cfg.TRAIN.CLASSES)

        print('classes:', classes)

        # read label image
        im_label = pad_im(cv2.imread(roidb['label'], cv2.IMREAD_UNCHANGED), 16)

        if roidb['flipped']:
            if len(im_label.shape) == 2:
                im_label = im_label[:, ::-1]
            else:
                im_label = im_label[:, ::-1, :]
        if im_scale != 1.0:
            im_label = cv2.resize(im_label, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_NEAREST)

        label_blob = np.zeros((num_classes, height, width), dtype=np.float32)
        label_blob[0, :, :] = 1.0
        for i in range(1, num_classes):
            I = np.where(im_label == classes[i])
            if len(I[0]) > 0:
                label_blob[i, I[0], I[1]] = 1.0
                label_blob[0, I[0], I[1]] = 0.0

        # foreground mask
        seg = torch.from_numpy((im_label != 0).astype(np.float32))
        mask = seg.unsqueeze(0).repeat((3, 1, 1)).float()

        # poses
        poses = meta_data['poses']
        if len(poses.shape) == 2:
            poses = np.reshape(poses, (3, 4, 1))
        if roidb['flipped']:
            poses = _flip_poses(poses, meta_data['intrinsic_matrix'], width)

        num = poses.shape[2]
        pose_blob = np.zeros((num_classes, 9), dtype=np.float32)
        gt_boxes = np.zeros((num_classes, 5), dtype=np.float32)
        count = 0
        for i in range(num):
            cls = int(meta_data['cls_indexes'][i])
            ind = np.where(classes == cls)[0]
            if len(ind) > 0:
                R = poses[:, :3, i]
                T = poses[:, 3, i]
                pose_blob[count, 0] = 1
                pose_blob[count, 1] = ind
                qt = mat2quat(R)

                # egocentric to allocentric
                qt_allocentric = egocentric2allocentric(qt, T)
                if qt_allocentric[0] < 0:
                   qt_allocentric = -1 * qt_allocentric
                pose_blob[count, 2:6] = qt_allocentric
                pose_blob[count, 6:] = T

                # compute box
                x3d = np.ones((4, self._points_all.shape[1]), dtype=np.float32)
                x3d[0, :] = self._points_all[ind,:,0]
                x3d[1, :] = self._points_all[ind,:,1]
                x3d[2, :] = self._points_all[ind,:,2]
                RT = np.zeros((3, 4), dtype=np.float32)
                RT[:3, :3] = quat2mat(qt)
                RT[:, 3] = T
                x2d = np.matmul(meta_data['intrinsic_matrix'], np.matmul(RT, x3d))
                x2d[0, :] = np.divide(x2d[0, :], x2d[2, :])
                x2d[1, :] = np.divide(x2d[1, :], x2d[2, :])
        
                gt_boxes[count, 0] = np.min(x2d[0, :]) * im_scale
                gt_boxes[count, 1] = np.min(x2d[1, :]) * im_scale
                gt_boxes[count, 2] = np.max(x2d[0, :]) * im_scale
                gt_boxes[count, 3] = np.max(x2d[1, :]) * im_scale
                gt_boxes[count, 4] = ind
                count += 1

        # construct the meta data
        """
        format of the meta_data
        intrinsic matrix: meta_data[0 ~ 8]
        inverse intrinsic matrix: meta_data[9 ~ 17]
        """
        K = np.matrix(meta_data['intrinsic_matrix']) * im_scale
        K[2, 2] = 1
        Kinv = np.linalg.pinv(K)
        meta_data_blob = np.zeros(18, dtype=np.float32)
        meta_data_blob[0:9] = K.flatten()
        meta_data_blob[9:18] = Kinv.flatten()

        # vertex regression target
        if cfg.TRAIN.VERTEX_REG:
            center = meta_data['center']
            if roidb['flipped']:
                center[:, 0] = width - center[:, 0]
            vertex_targets, vertex_weights = self._generate_vertex_targets(im_label,
                meta_data['cls_indexes'], center, poses, classes, num_classes)
        else:
            vertex_targets = []
            vertex_weights = []

        return label_blob, mask, meta_data_blob, pose_blob, gt_boxes, vertex_targets, vertex_weights

    def _get_default_path(self):
        """
        Return the default path where YCB_Video is expected to be installed.
        """
        return os.path.join(datasets.ROOT_DIR, 'data', 'YCB_Video')

    def _load_object_extents(self):

        extents = np.zeros((self._num_classes_all, 3), dtype=np.float32)
        for i in range(1, self._num_classes_all):
            point_file = os.path.join(self._model_path, self._classes_all[i], 'points.xyz')
            # print(point_file)
            assert os.path.exists(point_file), 'Path does not exist: {}'.format(point_file)
            points = np.loadtxt(point_file)
            extents[i, :] = 2 * np.max(np.absolute(points), axis=0)

        return extents

    def _load_object_points(self, classes, extents, symmetry):

        points = [[] for _ in range(len(classes))]
        num = np.inf
        num_classes = len(classes)
        for i in range(1, num_classes):
            point_file = os.path.join(self._model_path, classes[i], 'points.xyz')
            # print(point_file)
            assert os.path.exists(point_file), 'Path does not exist: {}'.format(point_file)
            points[i] = np.loadtxt(point_file)
            if points[i].shape[0] < num:
                num = points[i].shape[0]

        points_all = np.zeros((num_classes, num, 3), dtype=np.float32)
        for i in range(1, num_classes):
            points_all[i, :, :] = points[i][:num, :]

        # rescale the points
        point_blob = points_all.copy()
        for i in range(1, num_classes):
            # compute the rescaling factor for the points
            weight = 10.0 / np.max(extents[i, :])
            if weight < 10:
                weight = 10
            if symmetry[i] > 0:
                point_blob[i, :, :] = 4 * weight * point_blob[i, :, :]
            else:
                point_blob[i, :, :] = weight * point_blob[i, :, :]

        return points, points_all, point_blob

    def _load_image_set_index(self, image_set):
        """
        Load the indexes listed in this dataset's image set file.
        """
        image_set_file = os.path.join(self._ycb_video_path, image_set + '.txt')
        assert os.path.exists(image_set_file), \
                'Path does not exist: {}'.format(image_set_file)

        image_index = []
        video_ids_selected = set([])
        video_ids_not = set([])
        count = np.zeros((self.num_classes, ), dtype=np.int32)

        with open(image_set_file) as f:
            for x in f.readlines():
                index = x.rstrip('\n')
                pos = index.find('/')
                video_id = index[:pos]

                if not video_id in video_ids_selected and not video_id in video_ids_not:
                    filename = os.path.join(self._data_path, video_id, '000001-meta.mat')
                    meta_data = scipy.io.loadmat(filename)
                    print(meta_data['__header__'])
                    cls_indexes = meta_data['cls_indexes'].flatten()
                    flag = 0
                    for i in range(len(cls_indexes)):
                        cls_index = int(cls_indexes[i])
                        ind = np.where(np.array(cfg.TRAIN.CLASSES) == cls_index)[0]
                        if len(ind) > 0:
                            count[ind] += 1
                            flag = 1
                    if flag:
                        video_ids_selected.add(video_id)
                    else:
                        video_ids_not.add(video_id)

                if video_id in video_ids_selected:
                    image_index.append(index)

        for i in range(1, self.num_classes):
            print('%d %s [%d/%d]' % (i, self.classes[i], count[i], len(list(video_ids_selected))))

        # sample a subset for training
        if image_set == 'train':
            image_index = image_index[::5]

            # add synthetic data
            filename = os.path.join(self._data_path + '_syn', '*.mat')
            files = glob.glob(filename)
            print('adding synthetic %d data' % (len(files)))
            for i in range(len(files)):
                filename = files[i].replace(self._data_path, '../data')[:-9]
                image_index.append(filename)

        return image_index

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """

        prefix = '_class'
        for i in range(len(cfg.TRAIN.CLASSES)):
            prefix += '_%d' % cfg.TRAIN.CLASSES[i]
        cache_file = os.path.join(self.cache_path, self.name + prefix + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print('{} gt roidb loaded from {}'.format(self.name, cache_file))
            return roidb

        print('loading gt...')
        gt_roidb = [self._load_ycb_video_annotation(index)
                    for index in self._image_index]

        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print('wrote gt roidb to {}'.format(cache_file))

        return gt_roidb

    def _load_ycb_video_annotation(self, index):
        """
        Load class name and meta data
        """
        # image path
        image_path = self.image_path_from_index(index)

        # depth path
        depth_path = self.depth_path_from_index(index)

        # label path
        label_path = self.label_path_from_index(index)

        # metadata path
        metadata_path = self.metadata_path_from_index(index)

        # is synthetic image or not
        if 'data_syn' in image_path:
            is_syn = 1
            video_id = ''
            image_id = ''
        else:
            is_syn = 0
            # parse image name
            pos = index.find('/')
            video_id = index[:pos]
            image_id = index[pos+1:]
        
        return {'image': image_path,
                'depth': depth_path,
                'label': label_path,
                'meta_data': metadata_path,
                'video_id': video_id,
                'image_id': image_id,
                'is_syn': is_syn,
                'flipped': False}

    # image
    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self.image_index[i])

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """

        image_path = os.path.join(self._data_path, index + '-color.jpg')
        if not os.path.exists(image_path):
            image_path = os.path.join(self._data_path, index + '-color.png')

        assert os.path.exists(image_path), \
                'Path does not exist: {}'.format(image_path)
        return image_path

    # depth
    def depth_path_at(self, i):
        """
        Return the absolute path to depth i in the image sequence.
        """
        return self.depth_path_from_index(self.image_index[i])

    def depth_path_from_index(self, index):
        """
        Construct an depth path from the image's "index" identifier.
        """
        depth_path = os.path.join(self._data_path, index + '-depth.png')
        assert os.path.exists(depth_path), \
                'Path does not exist: {}'.format(depth_path)
        return depth_path

    # label
    def label_path_at(self, i):
        """
        Return the absolute path to metadata i in the image sequence.
        """
        return self.label_path_from_index(self.image_index[i])

    def label_path_from_index(self, index):
        """
        Construct an metadata path from the image's "index" identifier.
        """
        label_path = os.path.join(self._data_path, index + '-label.png')
        assert os.path.exists(label_path), \
                'Path does not exist: {}'.format(label_path)
        return label_path

    # camera pose
    def metadata_path_at(self, i):
        """
        Return the absolute path to metadata i in the image sequence.
        """
        return self.metadata_path_from_index(self.image_index[i])

    def metadata_path_from_index(self, index):
        """
        Construct an metadata path from the image's "index" identifier.
        """
        metadata_path = os.path.join(self._data_path, index + '-meta.mat')
        assert os.path.exists(metadata_path), \
                'Path does not exist: {}'.format(metadata_path)
        return metadata_path



