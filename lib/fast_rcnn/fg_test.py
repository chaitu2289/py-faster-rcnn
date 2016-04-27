# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Test a Fast R-CNN network on an imdb (image database)."""

from fast_rcnn.config import cfg, get_output_dir
from fast_rcnn.bbox_transform import clip_boxes, bbox_transform_inv
import argparse
from utils.timer import Timer
import numpy as np
import cv2
import caffe
from fast_rcnn.nms_wrapper import nms
import cPickle
import heapq
from utils.blob import im_list_to_blob
import os

import pdb
import xml.etree.ElementTree as ET

def _get_image_blob(im):
    """Converts an image into a network input.

    Arguments:
        im (ndarray): a color image in BGR order

    Returns:
        blob (ndarray): a data blob holding an image pyramid
        im_scale_factors (list): list of image scales (relative to im) used
            in the image pyramid
    """
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS

    im_shape = im_orig.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    processed_ims = []
    im_scale_factors = []

    for target_size in cfg.TEST.SCALES:
        im_scale = float(target_size) / float(im_size_min)
        # Prevent the biggest axis from being more than MAX_SIZE
        if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
            im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
        im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                        interpolation=cv2.INTER_LINEAR)
        im_scale_factors.append(im_scale)
        processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    return blob, np.array(im_scale_factors)

def _get_rois_blob(im_rois, im_scale_factors):
    """Converts RoIs into network inputs.

    Arguments:
        im_rois (ndarray): R x 4 matrix of RoIs in original image coordinates
        im_scale_factors (list): scale factors as returned by _get_image_blob

    Returns:
        blob (ndarray): R x 5 matrix of RoIs in the image pyramid
    """
    rois, levels = _project_im_rois(im_rois, im_scale_factors)
    rois_blob = np.hstack((levels, rois))
    return rois_blob.astype(np.float32, copy=False)

def _project_im_rois(im_rois, scales):
    """Project image RoIs into the image pyramid built by _get_image_blob.

    Arguments:
        im_rois (ndarray): R x 4 matrix of RoIs in original image coordinates
        scales (list): scale factors as returned by _get_image_blob

    Returns:
        rois (ndarray): R x 4 matrix of projected RoI coordinates
        levels (list): image pyramid levels used by each projected RoI
    """
    im_rois = im_rois.astype(np.float, copy=False)

    if len(scales) > 1:
        widths = im_rois[:, 2] - im_rois[:, 0] + 1
        heights = im_rois[:, 3] - im_rois[:, 1] + 1

        areas = widths * heights
        scaled_areas = areas[:, np.newaxis] * (scales[np.newaxis, :] ** 2)
        diff_areas = np.abs(scaled_areas - 224 * 224)
        levels = diff_areas.argmin(axis=1)[:, np.newaxis]
    else:
        levels = np.zeros((im_rois.shape[0], 1), dtype=np.int)

    rois = im_rois * scales[levels]

    return rois, levels

def _get_blobs(im, rois):
    """Convert an image and RoIs within that image into network inputs."""
    blobs = {'data' : None, 'rois' : None}
    blobs['data'], im_scale_factors = _get_image_blob(im)
    if not cfg.TEST.HAS_RPN:
        blobs['rois'] = _get_rois_blob(rois, im_scale_factors)
    return blobs, im_scale_factors

def im_detect(net, im, boxes=None, feature_generation=True):
    """Detect object classes in an image given object proposals.

    Arguments:
        net (caffe.Net): Fast R-CNN network to use
        im (ndarray): color image to test (in BGR order)
        boxes (ndarray): R x 4 array of object proposals or None (for RPN)

    Returns:
        scores (ndarray): R x K array of object class scores (K includes
            background as object category 0)
        boxes (ndarray): R x (4*K) array of predicted bounding boxes
    """
    boxes = np.append(boxes, np.array([[133, 169, 262, 316]]), axis=0)
    blobs, im_scales = _get_blobs(im, boxes)
    import pdb
    pdb.set_trace()
    ###
    #  Code added by chaitu
    ###
    if feature_generation:
    	blobs['rois_'] = np.array([np.append(0,box) for box in boxes])
	np.append(blobs['rois_'], np.array([0,133,169,262,316]))
    ###
    #  Code above added by chaitu
    ###

    # When mapping from image ROIs to feature map ROIs, there's some aliasing
    # (some distinct image ROIs get mapped to the same feature ROI).
    # Here, we identify duplicate feature ROIs, so we only compute features
    # on the unique subset.
    if cfg.DEDUP_BOXES > 0 and not cfg.TEST.HAS_RPN:
        v = np.array([1, 1e3, 1e6, 1e9, 1e12])
        hashes = np.round(blobs['rois'] * cfg.DEDUP_BOXES).dot(v)
        _, index, inv_index = np.unique(hashes, return_index=True,
                                        return_inverse=True)
        blobs['rois'] = blobs['rois'][index, :]
        boxes = boxes[index, :]

    if cfg.TEST.HAS_RPN:
        im_blob = blobs['data']
        blobs['im_info'] = np.array(
            [[im_blob.shape[2], im_blob.shape[3], im_scales[0]]],
            dtype=np.float32)


    # reshape network inputs
    net.blobs['data'].reshape(*(blobs['data'].shape))
    if cfg.TEST.HAS_RPN:
        net.blobs['im_info'].reshape(*(blobs['im_info'].shape))
	###code by chaitu
	if feature_generation:
		net.blobs['rois_'].reshape(*(blobs['rois_'].shape))
	###code above by chaitu
    else:
        net.blobs['rois'].reshape(*(blobs['rois'].shape))

    # do forward
    forward_kwargs = {'data': blobs['data'].astype(np.float32, copy=False)}
    if cfg.TEST.HAS_RPN:
        forward_kwargs['im_info'] = blobs['im_info'].astype(np.float32, copy=False)
	if feature_generation:
		forward_kwargs['rois_'] = blobs['rois_'].astype(np.float32, copy=False)
    else:
        forward_kwargs['rois'] = blobs['rois'].astype(np.float32, copy=False)
    #pdb.set_trace()
    blobs_out = net.forward(**forward_kwargs)
    import pdb
    pdb.set_trace()
    return net.blobs['fc7'].data, net.blobs['cls_prob'].data
    #pdb.set_trace()
    #if feature_generation:
    #	feature_vector = net.blobs['fc7'].data

    #if cfg.TEST.HAS_RPN:
    #    assert len(im_scales) == 1, "Only single-image batch implemented"
    #    rois = net.blobs['rois'].data.copy()
    #    # unscale back to raw image space
    #    boxes = rois[:, 1:5] / im_scales[0]

    #if cfg.TEST.SVM:
        # use the raw scores before softmax under the assumption they
        # were trained as linear SVMs
    #    scores = net.blobs['cls_score'].data
    #else:
        # use softmax estimated probabilities
    #    scores = blobs_out['cls_prob']

    #if cfg.TEST.BBOX_REG:
        # Apply bounding-box regression deltas
    #    box_deltas = blobs_out['bbox_pred']
    #    pred_boxes = bbox_transform_inv(boxes, box_deltas)
    #    pred_boxes = clip_boxes(pred_boxes, im.shape)
    #else:
        # Simply repeat the boxes, once for each class
    #    pred_boxes = np.tile(boxes, (1, scores.shape[1]))

    #if cfg.DEDUP_BOXES > 0 and not cfg.TEST.HAS_RPN:
        # Map scores and predictions back to the original set of boxes
    #    scores = scores[inv_index, :]
    #    pred_boxes = pred_boxes[inv_index, :]

    #return scores, pred_boxes

def vis_detections(im, class_name, dets, thresh=0.3):
    """Visual debugging of detections."""
    import matplotlib.pyplot as plt
    im = im[:, :, (2, 1, 0)]
    for i in xrange(np.minimum(10, dets.shape[0])):
        bbox = dets[i, :4]
        score = dets[i, -1]
        if score > thresh:
            plt.cla()
            plt.imshow(im)
            plt.gca().add_patch(
                plt.Rectangle((bbox[0], bbox[1]),
                              bbox[2] - bbox[0],
                              bbox[3] - bbox[1], fill=False,
                              edgecolor='g', linewidth=3)
                )
            plt.title('{}  {:.3f}'.format(class_name, score))
            plt.show()

def apply_nms(all_boxes, thresh):
    """Apply non-maximum suppression to all predicted boxes output by the
    test_net method.
    """
    num_classes = len(all_boxes)
    num_images = len(all_boxes[0])
    nms_boxes = [[[] for _ in xrange(num_images)]
                 for _ in xrange(num_classes)]
    for cls_ind in xrange(num_classes):
        for im_ind in xrange(num_images):
            dets = all_boxes[cls_ind][im_ind]
            if dets == []:
                continue
            keep = nms(dets, thresh)
            if len(keep) == 0:
                continue
            nms_boxes[cls_ind][im_ind] = dets[keep, :].copy()
    return nms_boxes

def test_net(net, imdb, image_id="003202"):
    """Test a Fast R-CNN network on an image database."""
    num_images = len(imdb.image_index)
    # heuristic: keep an average of 40 detections per class per images prior
    # to NMS
    max_per_set = 40 * num_images
    # heuristic: keep at most 100 detection per class per image prior to NMS
    max_per_image = 100
    # detection thresold for each class (this is adaptively set based on the
    # max_per_set constraint)
    thresh = -np.inf * np.ones(imdb.num_classes)
    # top_scores will hold one minheap of scores per class (used to enforce
    # the max_per_set constraint)
    top_scores = [[] for _ in xrange(imdb.num_classes)]
    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    all_boxes = [[[] for _ in xrange(num_images)]
                 for _ in xrange(imdb.num_classes)]

    output_dir = get_output_dir(imdb, net)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # timers
    _t = {'im_detect' : Timer(), 'misc' : Timer()}

    if not cfg.TEST.HAS_RPN:
        roidb = imdb.roidb

    if cfg.TEST.HAS_RPN:
    	box_proposals = None
    else:
      	box_proposals = roidb[i]['boxes'][roidb[i]['gt_classes'] == 0]
    ##
    # code below added by chaitu
    ##
    image_index_chaitu = image_id
    box_proposals = _load_pascal_annotation(image_index_chaitu) 
    #box_proposals['boxes'] = np.array([box_proposals['boxes'][0]])
    #box_proposals['gt_classes'] = np.array([box_proposals['gt_classes'][0]])
    box_proposals['boxes'] = box_proposals['boxes']
    box_proposals['gt_classes'] = box_proposals['gt_classes']
    gt_new_image = box_proposals['gt_classes'][0]
    ##
    # code above added by chaitu
    ##

    im = cv2.imread('/var/services/homes/kchakka/py-faster-rcnn/data/VOCdevkit2007/VOC2007/JPEGImages/'+image_index_chaitu+'.jpg')
    import pdb
    pdb.set_trace()
    feature_vector_, clas_prob = im_detect(net, im, box_proposals['boxes'])
    import pdb
    pdb.set_trace()
    feature_vector_ = feature_vector_.squeeze()
    feature_vector_ = feature_vector_/np.linalg.norm(feature_vector_)
    max_fv_ = 0
    output = []
    dot_prod_values = []
    #num_images = 10
    m = {}
    for i in xrange(num_images):
	print i
        # filter out any ground truth boxes
        if cfg.TEST.HAS_RPN:
            box_proposals = None
        else:
            box_proposals = roidb[i]['boxes'][roidb[i]['gt_classes'] == 0]
	##
	# code below added by chaitu
	##
	image_index_chaitu = imdb.image_path_at(i).split('/')[-1].split('.')[0] 
	m[i] = {}
	box_proposals = _load_pascal_annotation(image_index_chaitu) 
	##
	# code above added by chaitu
	##
        im = cv2.imread(imdb.image_path_at(i))
        _t['im_detect'].tic()
        fv_, class_prob = im_detect(net, im, box_proposals['boxes'])
        import pdb
	pdb.set_trace()
 	counter = 0
	similar_label = []
	non_similar_label = []
	import sys
	max_int = sys.maxint
	for fv_i in range(len(fv_)):
        #for f_ in fv_:
		f_ = fv_[fv_i]
		box = box_proposals['boxes'][fv_i]
		f_ = f_.squeeze()
		f_ = f_/np.linalg.norm(f_)
		dot_prod = np.dot(feature_vector_, f_)
		l2_distance = np.linalg.norm(feature_vector_ - f_)
		similar_image_index = [dot_prod, i, fv_i,box_proposals['gt_classes'][counter]]
		m[i][fv_i] = [box[0], box[1], box[2], box[3]]
		output.append(similar_image_index)
     
        _t['im_detect'].toc()
    output = sorted(output, reverse=True)
    #output = output[:20]
    train_txt = "/var/services/homes/kchakka/py-faster-rcnn/VOCdevkit/VOC2007/ImageSets/Main/train.txt"
    f = open(train_txt,'r')
    train_images = []
    similar_image_indices = []
    for line in f.readlines():
	train_images.append(line.strip())
    box_info = []
    for boxes in output:
	similar_image_indices.append(train_images[boxes[1]])
        box_info.append((train_images[boxes[1]], m[boxes[1]][boxes[2]]))
	boxes.append(int(train_images[boxes[1]]))
    print output[:20]
    print "Similar Image Indices : " , similar_image_indices[:20]

    import pdb
    pdb.set_trace()
    print "Images with box information : " , box_info
    for boxes in output:
	if boxes[3] == gt_new_image:
		if len(similar_label) <= 10:
			similar_label.append(boxes)
	else:
		if len(non_similar_label) <= 10:
			non_similar_label.append(boxes)
	if len(similar_label) > 10 and len(non_similar_label) > 10:
		break
    pdb.set_trace()	
    fine_tune_data = similar_label + non_similar_label
    print fine_tune_data
    f = open('finetune.txt', 'w')
    for label in fine_tune_data:
        id_length = len(str(label[4]))
	zeros = "0"*(6-id_length)
	f.write(zeros + str(label[4]) + "\n")
    f.close() 
     
    

        #_t['misc'].tic()
        #for j in xrange(1, imdb.num_classes):
        #    inds = np.where(scores[:, j] > thresh[j])[0]
        #    cls_scores = scores[inds, j]
        #    cls_boxes = boxes[inds, j*4:(j+1)*4]
        #    top_inds = np.argsort(-cls_scores)[:max_per_image]
        #    cls_scores = cls_scores[top_inds]
        #    cls_boxes = cls_boxes[top_inds, :]
        #    # push new scores onto the minheap
        #    for val in cls_scores:
        #        heapq.heappush(top_scores[j], val)
        #    # if we've collected more than the max number of detection,
        #    # then pop items off the minheap and update the class threshold
        #    if len(top_scores[j]) > max_per_set:
        #        while len(top_scores[j]) > max_per_set:
        #            heapq.heappop(top_scores[j])
        #        thresh[j] = top_scores[j][0]

        #    all_boxes[j][i] = \
        #            np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
        #            .astype(np.float32, copy=False)

        #    if 0:
        #        keep = nms(all_boxes[j][i], 0.3)
        #        vis_detections(im, imdb.classes[j], all_boxes[j][i][keep, :])
        #_t['misc'].toc()

        #print 'im_detect: {:d}/{:d} {:.3f}s {:.3f}s' \
        #      .format(i + 1, num_images, _t['im_detect'].average_time,
        #              _t['misc'].average_time)

    #for j in xrange(1, imdb.num_classes):
    #    for i in xrange(num_images):
    #        inds = np.where(all_boxes[j][i][:, -1] > thresh[j])[0]
    #        all_boxes[j][i] = all_boxes[j][i][inds, :]

    #det_file = os.path.join(output_dir, 'detections.pkl')
    #with open(det_file, 'wb') as f:
    #    cPickle.dump(all_boxes, f, cPickle.HIGHEST_PROTOCOL)

    #print 'Applying NMS to all detections'
    #nms_dets = apply_nms(all_boxes, cfg.TEST.NMS)
    #print 'Evaluating detections'
    #imdb.evaluate_detections(nms_dets, output_dir)



def _load_pascal_annotation(image_index):
        """
        Load image and bounding boxes info from XML file in the PASCAL VOC
        format.
        """
	#image_index = _load_image_set_index()
	classes = ('__background__', # always index 0
                         'aeroplane', 'bicycle', 'bird', 'boat',
                         'bottle', 'bus', 'car', 'cat', 'chair',
                         'cow', 'diningtable', 'dog', 'horse',
                         'motorbike', 'person', 'pottedplant',
                         'sheep', 'sofa', 'train', 'tvmonitor')
	num_classes = len(classes)
	_class_to_ind = dict(zip(classes, xrange(num_classes)))
	_data_path = "/var/services/homes/kchakka/py-faster-rcnn/VOCdevkit/VOC2007"
	image_index = [image_index]
		 
	for index in image_index:
        	filename = os.path.join(_data_path, 'Annotations', index + '.xml')
        	tree = ET.parse(filename)
        	objs = tree.findall('object')
        	if True:
            		# Exclude the samples labeled as difficult
            		non_diff_objs = [
                	obj for obj in objs if int(obj.find('difficult').text) == 0]
            		# if len(non_diff_objs) != len(objs):
            		#     print 'Removed {} difficult objects'.format(
            		#         len(objs) - len(non_diff_objs))
            		objs = non_diff_objs
       		num_objs = len(objs)

        	boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        	gt_classes = np.zeros((num_objs), dtype=np.int32)
		##
		#  commented below by chaitu
		##
        	#overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
		##
		#  commented above by chaitu
		##
        	# "Seg" area for pascal is just the box area
        	seg_areas = np.zeros((num_objs), dtype=np.float32)
	
        	# Load object bounding boxes into a data frame.
        	for ix, obj in enumerate(objs):
            		bbox = obj.find('bndbox')
            		# Make pixel indexes 0-based
            		x1 = float(bbox.find('xmin').text) - 1
            		y1 = float(bbox.find('ymin').text) - 1
            		x2 = float(bbox.find('xmax').text) - 1
            		y2 = float(bbox.find('ymax').text) - 1
            		cls = _class_to_ind[obj.find('name').text.lower().strip()]
            		boxes[ix, :] = [x1, y1, x2, y2]
            		gt_classes[ix] = cls
            		#overlaps[ix, cls] = 1.0
            		#seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)

        	#overlaps = scipy.sparse.csr_matrix(overlaps)
        return {'boxes' : boxes, 'gt_classes' : gt_classes}

def _load_image_set_index(_image_set="test"):
        """
        Load the indexes listed in this dataset's image set file.
        """
        # Example path to image set file:
        # self._devkit_path + /VOCdevkit2007/VOC2007/ImageSets/Main/val.txt
        _data_path = "/var/services/homes/kchakka/py-faster-rcnn/VOCdevkit/VOC2007"
        image_set_file = os.path.join(_data_path, 'ImageSets', 'Main',
                                      _image_set + '.txt')
        assert os.path.exists(image_set_file), \
                'Path does not exist: {}'.format(image_set_file)
        with open(image_set_file) as f:
            image_index = [x.strip() for x in f.readlines()]
        return image_index
