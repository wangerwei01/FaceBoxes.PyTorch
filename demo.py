from __future__ import print_function
import os
import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from data import cfg
from layers.functions.prior_box import PriorBox
from utils.nms_wrapper import nms
#from utils.nms.py_cpu_nms import py_cpu_nms
import cv2
import pdb
import math
from models.faceboxes import FaceBoxes
from utils.box_utils import decode

from utils.timer import Timer


parser = argparse.ArgumentParser(description='FaceBoxes')

parser.add_argument('-m', '--trained_model', default='./Final_FaceBoxes.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--save_folder', default='eval/', type=str, help='Dir to save results')
parser.add_argument('--cpu', action="store_true", default=False, help='Use cpu inference')
parser.add_argument('--dataset', default='./images', type=str, help='dataset')
parser.add_argument('--confidence_threshold', default=0.05, type=float, help='confidence_threshold')
parser.add_argument('--top_k', default=5000, type=int, help='top_k')
parser.add_argument('--nms_threshold', default=0.3, type=float, help='nms_threshold')
parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
args = parser.parse_args()



def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}




def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
#     check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model

def get_color(c, x, max_val):
    colors = torch.FloatTensor([[1, 0, 1], [0, 0, 1], [0, 1, 1], [0, 1, 0],
                                [1, 1, 0], [1, 0, 0]])
    ratio = float(x) / max_val * 5
    i = int(math.floor(ratio))
    j = int(math.ceil(ratio))
    ratio = ratio - i
    r = (1 - ratio) * colors[i][c] + ratio * colors[j][c]
    return int(r * 255)


def draw_rects_use_boxes(img, rects, classes):
    for cls_id in range(1,len(classes)):
        boxes = rects[cls_id]
        if boxes.shape[0] == 0:
            continue
        for idx in range(len(boxes)):
            rect = boxes[idx]
            if(rect[-1]<0.6):
                continue
            left_top = (int(float(rect[0])), int(float(rect[1])))
            right_bottom = (int(float(rect[2])), int(float(rect[3])))
            sroce = float(rect[-1])
            label = "{0}".format(classes[cls_id])
            class_len = len(classes)
            offset = cls_id * 123457 % class_len
            red = get_color(2, offset, class_len)
            green = get_color(1, offset, class_len)
            blue = get_color(0, offset, class_len)
            color = (blue, green, red)
            cv2.rectangle(img, left_top, right_bottom, color, 1)
            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
            right_bottom = left_top[0] + t_size[0] +60, left_top[1] - t_size[1] - 10
            cv2.rectangle(img, left_top, right_bottom, color, -1)
            cv2.putText(img,
                        str(label)+':'+'%.2f'%sroce,
                        (left_top[0], left_top[1] - t_size[1] +8),
                        cv2.FONT_HERSHEY_PLAIN, 1, [225, 255, 255], 1)
    return img


className = {0:"background",1:"face"}



if __name__ == '__main__':
    torch.set_grad_enabled(False)
    net = FaceBoxes(phase='test', size=None, num_classes=2)    # initialize detector
    net = load_model(net, args.trained_model, args.cpu)
    net.eval()
    print('Finished loading model!')
    print(net)
    cudnn.benchmark = True
    device = torch.device("cpu" if args.cpu else "cuda:0")
    net = net.to(device)
    
    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)
    
    image_file = "/workspace/mnt/cache/wangerwei/FaceDataset/FDDB_Data/FaceDetectionTrainValData/widerface"
    fim = open(os.path.join(image_file,'ImageSets/Main/test.txt'),'r')
    lines = fim.readlines()
    
    #demo_images = os.listdir(args.dataset)
    
    num_images = len(lines)
    
    resize=1
    _t = {'forward_pass': Timer(), 'misc': Timer()}
    
    for idx,line in enumerate(lines):
        print(line)
        img = np.float32(cv2.imread(os.path.join(image_file,'JPEGImages/'+line.strip('\n')+'.jpg'),cv2.IMREAD_COLOR))
        #print(img.shape)
        #img = cv2.resize(img,(512,512), interpolation=cv2.INTER_LINEAR)
        #img = cv2.resize(img, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)
        origin_img = img.copy()
        
        im_height, im_width, _ = img.shape
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(device)
        scale = scale.to(device)
        
        _t['forward_pass'].tic()
        loc, conf = net(img)  # forward pass
        loc = loc.cpu()
        conf= conf.cpu()
        _t['forward_pass'].toc()
        loc = loc.cuda()
        conf= conf.cuda()
        _t['misc'].tic()
        
        priorbox = PriorBox(cfg, image_size=(im_height, im_width))
        priors = priorbox.forward()
        priors = priors.to(device)
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
        boxes = boxes * scale / 1
        boxes = boxes.cpu().numpy()
        scores = conf.data.cpu().numpy()[:, 1]

        # ignore low scores
        inds = np.where(scores > args.confidence_threshold)[0]
        boxes = boxes[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1][:args.top_k]
        boxes = boxes[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        #keep = py_cpu_nms(dets, args.nms_threshold)
        keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
        dets = dets[keep, :]

        # keep top-K faster NMS
        dets = dets[:args.keep_top_k, :]
        _t['misc'].toc()
        print('im_detect: {:d}/{:d} forward_pass_time: {:.4f}s misc: {:.4f}s'.format(idx + 1, num_images, _t['forward_pass'].average_time, _t['misc'].average_time))
        
        
        
#         fw = open(os.path.join(args.save_folder,line.strip('\n')+'.txt'), 'a')
#         fw.write(line.strip('\n')+'.jpg\n')
#         fw.write(str(dets.shape[0])+'\n')
        
        
#         for k in range(dets.shape[0]):
# #                 if(dets[k,-1]<0.6):
# #                     continue
#                 xmin = dets[k, 0]
#                 ymin = dets[k, 1]
#                 xmax = dets[k, 2]
#                 ymax = dets[k, 3]
#                 score = dets[k, 4]
#                 fw.write('{:d},{:d},{:d},{:d},{:.6f}\n'.format(int(xmin), int(ymin), int(xmax-xmin), int(ymax-ymin),score))
#         fw.close()
        
        if True:
            ds = {}
            ds[0]=np.empty((0,5))
            ds[1] = np.empty((0, 5))
            for ind,bbox in enumerate(dets):
                ds[1] = np.vstack((ds[1], bbox))
            imm = draw_rects_use_boxes(origin_img,ds,className)
            if not os.path.exists("./debug"):
                os.makedirs("./debug")
            image_path = os.path.join("./debug",line)
            cv2.imwrite(image_path,imm)
        


    