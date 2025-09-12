import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from random import randint
import numpy as np
import cv2
from PIL import Image
import random
import os
import math

###################################################################
# random mask generation
###################################################################


def random_regular_mask(img, ratio=None, min_ratio=0.2, max_ratio=0.6):
    """Generates a random regular hole"""
    s = img.size()
    mask = torch.ones(1, s[1], s[2])

    if ratio is None:
        ratio = random.uniform(min_ratio, max_ratio)
    else:
        ratio = max_ratio if ratio>max_ratio else ratio
        ratio = min_ratio if ratio<min_ratio else ratio


    start = 1 + int((ratio-min_ratio)*8)
    end = start + 1


    N_mask = random.randint(start, end)

    if ratio is not None:
        total_mask_area = ratio * s[1] * s[2]
        target_area = total_mask_area / N_mask 


    limx = s[1] - s[1] / (N_mask + 1)
    limy = s[2] - s[2] / (N_mask + 1)
    for _ in range(N_mask):
        x = random.randint(0, int(limx))
        y = random.randint(0, int(limy))

        if ratio is not None:
            min_size_x = max(1, int(math.sqrt(target_area))-20)
            min_size_y = max(1, int(math.sqrt(target_area))-20)

            max_size_x = min_size_x + int(s[1]/((max_ratio*10)+1-N_mask))
            max_size_y = min_size_y + int(s[2]/((max_ratio*10)+1-N_mask))

            range_x = x + random.randint(min_size_x, max_size_x)
            range_y = y + random.randint(min_size_y, max_size_y)
        else:
            range_x = x + random.randint(int(s[1] / (N_mask + 1)), int(s[1] - x))
            range_y = y + random.randint(int(s[2] / (N_mask + 1)), int(s[2] - y))
        mask[:, int(x):int(range_x), int(y):int(range_y)] = 0
    return mask


def center_mask(img, ratio=None, min_ratio=0.2, max_ratio=0.6):
    """Generates a center hole with 1/4*W and 1/4*H"""

    size = img.size()
    mask = torch.ones(1, size[1], size[2])


    if ratio is None:
        ratio = random.uniform(min_ratio, max_ratio)*0.6

    if ratio is not None:
        ratio = max(min_ratio, min(max_ratio, ratio))
        mask_width = int(size[1]*math.sqrt(ratio))
        mask_height = int(size[2]*math.sqrt(ratio))

        x = int((size[1] - mask_width) / 2)
        y = int((size[2] - mask_height) / 2)
        

        range_x = x + mask_width
        range_y = y + mask_height

    else:
        x = int(size[1] / 4)
        y = int(size[2] / 4)
        range_x = int(size[1] * 3 / 4)
        range_y = int(size[2] * 3 / 4)


    mask[:, x:range_x, y:range_y] = 0

    return mask


def random_irregular_mask(img, ratio=None, min_ratio=0.2, max_ratio=0.6):
    """Generates a random irregular mask with lines, circles and elipses"""
    transform = transforms.Compose([transforms.ToTensor()])

    size = img.size()
    mask = torch.ones(1, size[1], size[2])
    img = np.zeros((size[1], size[2], 1), np.uint8)

    if ratio is None:
        ratio = random.uniform(min_ratio, max_ratio)
    else:
        ratio = max_ratio if ratio>max_ratio else ratio
        ratio = min_ratio if ratio<min_ratio else ratio

    # Set size scale
    max_width = 20
    if size[1] < 64 or size[2] < 64:
        raise Exception("Width and Height of mask must be at least 64!")

    if ratio is not None:
        start = 12 + int((ratio-min_ratio)*60)
        end = 16 + int((ratio-min_ratio)*60)
    else:
        start = 12
        end = 48
    number = random.randint(start, end)
    for _ in range(number):
        model = random.random()
        if model < 0.6:
            # Draw random lines
            x1, x2 = randint(1, size[1]), randint(1, size[1])
            y1, y2 = randint(1, size[2]), randint(1, size[2])
            thickness = randint(4, max_width)
            cv2.line(img, (x1, y1), (x2, y2), (1, 1, 1), thickness)

        elif model > 0.6 and model < 0.8:
            # Draw random circles
            x1, y1 = randint(1, size[1]), randint(1, size[2])
            radius = randint(4, max_width)
            cv2.circle(img, (x1, y1), radius, (1, 1, 1), -1)

        elif model > 0.8:
            # Draw random ellipses
            x1, y1 = randint(1, size[1]), randint(1, size[2])
            s1, s2 = randint(1, size[1]), randint(1, size[2])
            a1, a2, a3 = randint(3, 180), randint(3, 180), randint(3, 180)
            thickness = randint(4, max_width)
            cv2.ellipse(img, (x1, y1), (s1, s2), a1, a2, a3, (1, 1, 1), thickness)

    img = img.reshape(size[2], size[1])
    img = Image.fromarray(img*255)

    img_mask = transform(img)
    mask[0, :, :] = img_mask < 1

    return mask


def random_freefrom_mask(img, ratio=None, mv=3, ma=4.0, ml=40, mbw=10, min_ratio=0.2, max_ratio=0.6):
    transform = transforms.Compose([transforms.ToTensor()])
    size = img.size()
    mask = torch.ones(1, size[1], size[2])
    img = np.zeros((size[1], size[2],1), np.uint8)

    if ratio is None:
        ratio = random.uniform(min_ratio, max_ratio)


    ratio = min_ratio if ratio<min_ratio else ratio
    ratio = max_ratio if ratio>max_ratio else ratio
    start = int(12+(ratio-0.2)*80)


    num_v = start + np.random.randint(mv)  # tf.random_uniform([], minval=0, maxval=config.MAXVERTEX, dtype=tf.int32)
    for i in range(num_v):
        start_x = np.random.randint(size[1])
        start_y = np.random.randint(size[2])
        for j in range(1 + np.random.randint(5)):
            angle = 0.01 + np.random.randint(ma)
            if i % 2 == 0:
                angle = 2 * 3.1415926 - angle
            length = 10 + np.random.randint(ml)
            brush_w = 10 + np.random.randint(mbw)
            end_x = (start_x + length * np.sin(angle)).astype(np.int32)
            end_y = (start_y + length * np.cos(angle)).astype(np.int32)

            cv2.line(img, (start_y, start_x), (end_y, end_x), 1.0, brush_w)
            start_x, start_y = end_x, end_y


    img = img.reshape(size[2], size[1])
    img = Image.fromarray(img*255)

    img_mask = transform(img)
    mask[0, :, :] = img_mask < 1

    return mask

###################################################################
# multi scale for image generation
###################################################################


def scale_img(img, size):
    scaled_img = F.interpolate(img, size=size, mode='bilinear', align_corners=True)
    return scaled_img


def scale_pyramid(img, num_scales):
    scaled_imgs = [img]

    s = img.size()

    h = s[2]
    w = s[3]

    for i in range(1, num_scales):
        ratio = 2**i
        nh = h // ratio
        nw = w // ratio
        scaled_img = scale_img(img, size=[nh, nw])
        scaled_imgs.append(scaled_img)

    scaled_imgs.reverse()
    return scaled_imgs



def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) 

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# generate fix masks
def sample_final_mask_sets(save_dir, regular=True, ratio='small', center=False, dataset='celebaq'):

    set_random_seed(42)

    # celebaq mask sets 2000 validation
    if dataset == 'celebaq':
        num = 2000
    elif dataset == 'places2':
        num = 36500

    img = torch.rand(3,256,256)
    save_dir = os.path.join(save_dir, dataset)

    
    if regular:
        postidx='regular'
    else:
        if ratio == 'small':
            postidx = 'irregular_small'
        elif ratio == 'large':
            postidx = 'irregular_large'

    save_dir = os.path.join(save_dir, postidx)
    os.makedirs(save_dir, exist_ok=True)


    if regular:
        min_ratio=0.2
        max_ratio=0.5
        
        for i in range(num):
            idx = torch.randint(0, 2, size=(1,)).item()
            if idx==0:
                mask = random_regular_mask(img, ratio=None, min_ratio=min_ratio, max_ratio=max_ratio)*255
            elif idx==1:
                mask = center_mask(img, ratio=None, min_ratio=min_ratio, max_ratio=max_ratio)*255
            path = os.path.join(save_dir, f'regular_{i}.jpg')
            cv2.imwrite(path, mask.permute(1,2,0).numpy())
    
    else:
        if ratio=='small':
            min_ratio = 0.2
            max_ratio = 0.4
        elif ratio=='large':
            min_ratio=0.4
            max_ratio=0.6

        for i in range(num):
            model_idx = torch.randint(0,2, size=(1,)).item()
            idx = torch.randint(0,4, size=(1,)).item()
            if model_idx == 0:
                mask = random_irregular_mask(img, ratio=None, min_ratio=min_ratio, max_ratio=max_ratio)*255
            elif model_idx == 1:
                mask = random_freefrom_mask(img, ratio=None, min_ratio=min_ratio, max_ratio=max_ratio)*255
            path = os.path.join(save_dir, f'irregular_{ratio}_{i}.jpg')
            cv2.imwrite(path, mask.permute(1,2,0).numpy())
