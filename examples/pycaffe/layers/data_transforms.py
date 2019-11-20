# -*- coding: utf-8 -*-
import types

import cv2
import numpy as np
from numpy import random


def intersect(box_a, box_b):
    max_xy = np.minimum(box_a[:, 2:], box_b[2:])
    min_xy = np.maximum(box_a[:, :2], box_b[:2])
    inter = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf)
    return inter[:, 0] * inter[:, 1]


def jaccard_numpy(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: Multiple bounding boxes, Shape: [num_boxes,4]
        box_b: Single bounding box, Shape: [4]
    Return:
        jaccard overlap: Shape: [box_a.shape[0], box_a.shape[1]]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2] - box_a[:, 0]) *
              (box_a[:, 3] - box_a[:, 1]))  # [A,B]
    area_b = ((box_b[2] - box_b[0]) *
              (box_b[3] - box_b[1]))  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


def object_converage_numpy(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: Multiple bounding boxes, Shape: [num_boxes,4]
        box_b: Single bounding box, Shape: [4]
    Return:
        jaccard overlap: Shape: [box_a.shape[0], box_a.shape[1]]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2] - box_a[:, 0]) *
              (box_a[:, 3] - box_a[:, 1]))  # [A,B]
    area_b = ((box_b[2] - box_b[0]) *
              (box_b[3] - box_b[1]))  # [A,B]
    return inter / area_a  # [A,B]


class Compose(object):
    """Composes several augmentations together.
    Args:
        transforms (List[Transform]): list of transforms to compose.
    Example:
        augmentations.Compose([
            transforms.CenterCrop(10),
            transforms.ToTensor(),
        ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, boxes=None, labels=None, landmarks=None):
        for t in self.transforms:
            img, boxes, labels, landmarks = t(img, boxes, labels, landmarks)
        return img, boxes, labels, landmarks


class Lambda(object):
    """Applies a lambda as a transform."""

    def __init__(self, lambd):
        assert isinstance(lambd, types.LambdaType)
        self.lambd = lambd

    def __call__(self, img, boxes=None, labels=None):
        return self.lambd(img, boxes, labels)


class ConvertFromInts(object):
    def __call__(self, image, boxes=None, labels=None, landmarks=None):
        return image.astype(np.float32), boxes, labels, landmarks


class SubtractMeans(object):
    def __init__(self, mean):
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, image, boxes=None, labels=None, landmarks=None):
        image = image.astype(np.float32)
        image -= self.mean
        return image.astype(np.float32), boxes, labels, landmarks


class DivStd(object):
    def __init__(self, std):
        self.std = np.array(std, dtype=np.float32)

    def __call__(self, image, boxes=None, labels=None, landmarks=None):
        image = image.astype(np.float32)
        image /= self.std
        return image.astype(np.float32), boxes, labels, landmarks


class ChannelFirst(object):
    def __call__(self, image, boxes=None, labels=None, landmarks=None):
        return np.transpose(image, (2, 0, 1)), boxes, labels, landmarks


class imgprocess(object):
    def __init__(self, std):
        self.std = np.array(std, dtype=np.float32)

    def __call__(self, image, boxes=None, labels=None):
        image = image.astype(np.float32)
        image /= self.std
        return image.astype(np.float32), boxes, labels


class ToAbsoluteCoords(object):
    def __call__(self, image, boxes=None, labels=None):
        height, width, channels = image.shape
        boxes[:, 0] *= width
        boxes[:, 2] *= width
        boxes[:, 1] *= height
        boxes[:, 3] *= height

        return image, boxes, labels


class ToPercentCoords(object):
    def __call__(self, image, boxes=None, labels=None, landmarks=None):
        height, width, channels = image.shape
        boxes[:, 0] /= width
        boxes[:, 2] /= width
        boxes[:, 1] /= height
        boxes[:, 3] /= height

        landmarks[:, 0::2] /= width
        landmarks[:, 1::2] /= height

        return image, boxes, labels, landmarks


class Resize(object):
    def __init__(self, size=(300, 300)):
        self.size = size

    def __call__(self, image, boxes=None, labels=None, landmarks=None):
        image = cv2.resize(image, (self.size[0],
                                   self.size[1]))
        return image, boxes, labels, landmarks


class RandomSaturation(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, image, boxes=None, labels=None, landmarks=None):
        if random.randint(2):
            image[:, :, 1] *= random.uniform(self.lower, self.upper)

        return image, boxes, labels, landmarks


class RandomHue(object):
    def __init__(self, delta=18.0):
        assert delta >= 0.0 and delta <= 360.0
        self.delta = delta

    def __call__(self, image, boxes=None, labels=None, landmarks=None):
        if random.randint(2):
            image[:, :, 0] += random.uniform(-self.delta, self.delta)
            image[:, :, 0][image[:, :, 0] > 360.0] -= 360.0
            image[:, :, 0][image[:, :, 0] < 0.0] += 360.0
        return image, boxes, labels, landmarks


class RandomLightingNoise(object):
    def __init__(self):
        self.perms = ((0, 1, 2), (0, 2, 1),
                      (1, 0, 2), (1, 2, 0),
                      (2, 0, 1), (2, 1, 0))

    def __call__(self, image, boxes=None, labels=None, landmarks=None):
        if random.randint(2):
            swap = self.perms[random.randint(len(self.perms))]
            shuffle = SwapChannels(swap)  # shuffle channels
            image = shuffle(image)
        return image, boxes, labels, landmarks


class ConvertColor(object):
    def __init__(self, current, transform):
        self.transform = transform
        self.current = current

    def __call__(self, image, boxes=None, labels=None, landmarks=None):
        if self.current == 'BGR' and self.transform == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif self.current == 'RGB' and self.transform == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        elif self.current == 'BGR' and self.transform == 'RGB':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif self.current == 'HSV' and self.transform == 'BGR':
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        elif self.current == 'HSV' and self.transform == "RGB":
            image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
        else:
            raise NotImplementedError
        return image, boxes, labels, landmarks


class RandomContrast(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    # expects float image
    def __call__(self, image, boxes=None, labels=None, landmarks=None):
        if random.randint(2):
            alpha = random.uniform(self.lower, self.upper)
            image *= alpha
        return image, boxes, labels, landmarks


class RandomBrightness(object):
    def __init__(self, delta=32):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def __call__(self, image, boxes=None, labels=None, landmarks=None):
        if random.randint(2):
            delta = random.uniform(-self.delta, self.delta)
            image += delta
        return image, boxes, labels, landmarks


class ToCV2Image(object):
    def __call__(self, tensor, boxes=None, labels=None):
        return tensor.cpu().numpy().astype(np.float32).transpose((1, 2, 0)), boxes, labels


class RandomSampleCrop(object):
    """Crop
    Arguments:
        img (Image): the image being input during training
        boxes (Tensor): the original bounding boxes in pt form
        labels (Tensor): the class labels for each bbox
        mode (float tuple): the min and max jaccard overlaps
    Return:
        (img, boxes, classes)
            img (Image): the cropped image
            boxes (Tensor): the adjusted bounding boxes in pt form
            labels (Tensor): the class labels for each bbox
    """

    def __init__(self):
        self.sample_options = (
            # using entire original input image
            None,
            # sample a patch s.t. MIN jaccard w/ obj in .1,.3,.4,.7,.9
            (0.1, None),
            (0.3, None),
            (0.7, None),
            (0.9, None),
            # randomly sample a patch
            (None, None),
        )

    def __call__(self, image, boxes=None, labels=None, landmarks=None):
        height, width, _ = image.shape
        while True:
            # randomly choose a mode
            mode = random.choice(self.sample_options)
            if mode is None:
                return image, boxes, labels, landmarks

            min_iou, max_iou = mode
            if min_iou is None:
                min_iou = float('-inf')
            if max_iou is None:
                max_iou = float('inf')

            # max trails (50)
            for _ in range(50):
                current_image = image

                w = random.uniform(0.3 * width, width)
                h = random.uniform(0.3 * height, height)

                # aspect ratio constraint b/t .5 & 2
                if h / w < 0.5 or h / w > 2:
                    continue

                left = random.uniform(width - w)
                top = random.uniform(height - h)

                # convert to integer rect x1,y1,x2,y2
                rect = np.array([int(left), int(top), int(left + w), int(top + h)])

                # calculate IoU (jaccard overlap) b/t the cropped and gt boxes
                overlap = jaccard_numpy(boxes, rect)

                # is min and max overlap constraint satisfied? if not try again
                if overlap.max() < min_iou or overlap.min() > max_iou:
                    continue

                # cut the crop from the image
                current_image = current_image[rect[1]:rect[3], rect[0]:rect[2], :]

                # keep overlap with gt box IF center in sampled patch
                centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0

                # mask in all gt boxes that above and to the left of centers
                m1 = (rect[0] < centers[:, 0]) * (rect[1] < centers[:, 1])

                # mask in all gt boxes that under and to the right of centers
                m2 = (rect[2] > centers[:, 0]) * (rect[3] > centers[:, 1])

                # mask in that both m1 and m2 are true
                mask = m1 * m2

                # have any valid boxes? try again if not
                if not mask.any():
                    continue

                # take only matching gt boxes
                current_boxes = boxes[mask, :].copy()

                # take only matching gt labels
                current_labels = labels[mask]

                current_landmarks = landmarks[mask, :].copy()

                # should we use the box left and top corner or the crop's
                current_boxes[:, :2] = np.maximum(current_boxes[:, :2], rect[:2])
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, :2] -= rect[:2]

                current_boxes[:, 2:] = np.minimum(current_boxes[:, 2:], rect[2:])
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, 2:] -= rect[:2]

                for i in range(int(current_landmarks.shape[-1] / 2)):
                    current_landmarks[:, 2 * i:2 * (i + 1)] = np.maximum(current_landmarks[:, 2 * i:2 * (i + 1)],
                                                                         rect[:2])
                    current_landmarks[:, 2 * i:2 * (i + 1)] = np.minimum(current_landmarks[:, 2 * i:2 * (i + 1)],
                                                                         rect[2:])
                    current_landmarks[:, 2 * i:2 * (i + 1)] -= rect[:2]

                return current_image, current_boxes, current_labels, current_landmarks


class RandomSampleCrop_v2(object):
    """Crop
    Arguments:
        img (Image): the image being input during training
        boxes (Tensor): the original bounding boxes in pt form
        labels (Tensor): the class labels for each bbox
        mode (float tuple): the min and max jaccard overlaps
    Return:
        (img, boxes, classes)
            img (Image): the cropped image
            boxes (Tensor): the adjusted bounding boxes in pt form
            labels (Tensor): the class labels for each bbox
    """

    def __init__(self):
        self.sample_options = (
            # using entire original input image
            None,
            # sample a patch s.t. MIN jaccard w/ obj in .1,.3,.4,.7,.9

            # randomly sample a patch
            (1, None),
            (1, None),
            (1, None),
            (1, None),
        )

    def __call__(self, image, boxes=None, labels=None, landmarks=None):
        height, width, _ = image.shape
        while True:
            # randomly choose a mode
            mode = random.choice(self.sample_options)
            if mode is None:
                return image, boxes, labels, landmarks

            min_iou, max_iou = mode
            if min_iou is None:
                min_iou = float('-inf')
            if max_iou is None:
                max_iou = float('inf')

            # max trails (50)
            for _ in range(50):
                current_image = image

                w = random.uniform(0.3 * width, width)
                h = random.uniform(0.3 * height, height)

                # aspect ratio constraint b/t .5 & 2
                if h / w != 1:
                    continue
                print("1")
                left = random.uniform(width - w)
                top = random.uniform(height - h)

                # convert to integer rect x1,y1,x2,y2
                rect = np.array([int(left), int(top), int(left + w), int(top + h)])

                # calculate IoU (jaccard overlap) b/t the cropped and gt boxes
                overlap = object_converage_numpy(boxes, rect)

                # is min and max overlap constraint satisfied? if not try again
                if overlap.max() < min_iou or overlap.min() > max_iou:
                    continue

                # cut the crop from the image
                current_image = current_image[rect[1]:rect[3], rect[0]:rect[2], :]

                # keep overlap with gt box IF center in sampled patch
                centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0

                # mask in all gt boxes that above and to the left of centers
                m1 = (rect[0] < centers[:, 0]) * (rect[1] < centers[:, 1])

                # mask in all gt boxes that under and to the right of centers
                m2 = (rect[2] > centers[:, 0]) * (rect[3] > centers[:, 1])

                # mask in that both m1 and m2 are true
                mask = m1 * m2

                # have any valid boxes? try again if not
                if not mask.any():
                    continue

                # take only matching gt boxes
                current_boxes = boxes[mask, :].copy()

                # take only matching gt labels
                current_labels = labels[mask]

                current_landmarks = landmarks[mask]

                # should we use the box left and top corner or the crop's
                current_boxes[:, :2] = np.maximum(current_boxes[:, :2], rect[:2])
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, :2] -= rect[:2]

                current_boxes[:, 2:] = np.minimum(current_boxes[:, 2:], rect[2:])
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, 2:] -= rect[:2]

                for i in range(int(current_landmarks.shape[-1] / 2)):
                    current_landmarks[:, 2 * i:2 * (i + 1)] = np.maximum(current_landmarks[:, 2 * i:2 * (i + 1)],
                                                                         rect[:2])
                    current_landmarks[:, 2 * i:2 * (i + 1)] = np.minimum(current_landmarks[:, 2 * i:2 * (i + 1)],
                                                                         rect[2:])
                    current_landmarks[:, 2 * i:2 * (i + 1)] -= rect[:2]

                return current_image, current_boxes, current_labels, current_landmarks


class Expand(object):
    def __init__(self, mean):
        self.mean = mean

    def __call__(self, image, boxes, labels):
        if random.randint(2):
            return image, boxes, labels

        height, width, depth = image.shape
        ratio = random.uniform(1, 4)
        left = random.uniform(0, width * ratio - width)
        top = random.uniform(0, height * ratio - height)

        expand_image = np.zeros(
            (int(height * ratio), int(width * ratio), depth),
            dtype=image.dtype)
        expand_image[:, :, :] = self.mean
        expand_image[int(top):int(top + height),
        int(left):int(left + width)] = image
        image = expand_image

        boxes = boxes.copy()
        boxes[:, :2] += (int(left), int(top))
        boxes[:, 2:] += (int(left), int(top))

        return image, boxes, labels


class RandomMirror(object):
    def __call__(self, image, boxes, classes, landmarks):
        _, width, _ = image.shape
        if random.randint(2):
            image = image[:, ::-1]
            boxes = boxes.copy()
            boxes[:, 0::2] = width - boxes[:, 2::-2]  # 这里是对的, 不是boxes[:, 0::2] = width - boxes[:, 0::2]
            landmarks = landmarks.copy()
            landmarks[:, 0::2] = width - landmarks[:, 0::2]
            # 关键点的顺序将会变化, 1 <-> 2, 4 <-> 5
            temp = landmarks[:, 0:2].copy()
            landmarks[:, 0:2] = landmarks[:, 2:4].copy()
            landmarks[:, 2:4] = temp.copy()
            temp = landmarks[:, 6:8].copy()
            landmarks[:, 6:8] = landmarks[:, 8:10].copy()
            landmarks[:, 8:10] = temp.copy()
        return image, boxes, classes, landmarks


class RandomRotate(object):
    def __init__(self, max_degree=15):
        self.max_degree = max(max_degree, 15)

    def __call__(self, image, boxes, classes, landmarks):
        if random.randint(2):
            degree = random.uniform(-self.max_degree, self.max_degree)
            height, width, _ = image.shape
            img_center = (width / 2.0, height / 2.0)
            rotateMat = cv2.getRotationMatrix2D(img_center, degree, 1.0)

            # image
            rot_img = cv2.warpAffine(image, rotateMat, (width, height), flags=cv2.INTER_CUBIC,
                                     borderMode=cv2.BORDER_CONSTANT)

            # box
            new_boxes = boxes.copy()
            for i in range(len(boxes)):
                new_x1, new_y1 = rotateMat.dot(np.array([boxes[i][0], boxes[i][1], 1]))
                new_x2, new_y2 = rotateMat.dot(np.array([boxes[i][2], boxes[i][1], 1]))
                new_x3, new_y3 = rotateMat.dot(np.array([boxes[i][0], boxes[i][3], 1]))
                new_x4, new_y4 = rotateMat.dot(np.array([boxes[i][2], boxes[i][3], 1]))
                new_boxes[i][0] = min(width, max(0, min([new_x1, new_x2, new_x3, new_x4])))
                new_boxes[i][1] = min(height, max(0, min([new_y1, new_y2, new_y3, new_y4])))
                new_boxes[i][2] = min(width, max(0, max([new_x1, new_x2, new_x3, new_x4])))
                new_boxes[i][3] = min(height, max(0, max([new_y1, new_y2, new_y3, new_y4])))

            # key points
            new_landmarks = landmarks.copy()
            for i in range(len(landmarks)):
                if sum(landmarks[i]) == 0:
                    continue
                for j in range(0, 10, 2):
                    new_landmarks[i][j], new_landmarks[i][j + 1] = rotateMat.dot(
                        np.array([landmarks[i][j], landmarks[i][j + 1], 1]))
                    if new_landmarks[i][j] < 0 or new_landmarks[i][j] > width or new_landmarks[i][j + 1] < 0 or \
                            new_landmarks[i][j + 1] > height:
                        new_landmarks[i, :] = 0
                        break
            return rot_img, new_boxes, classes, new_landmarks
        return image, boxes, classes, landmarks


class SwapChannels(object):
    """Transforms a tensorized image by swapping the channels in the order
     specified in the swap tuple.
    Args:
        swaps (int triple): final order of channels
            eg: (2, 1, 0)
    """

    def __init__(self, swaps):
        self.swaps = swaps

    def __call__(self, image):
        """
        Args:
            image (Tensor): image tensor to be transformed
        Return:
            a tensor with channels swapped according to swap
        """
        image = image[:, :, self.swaps]
        return image


class PhotometricDistort(object):
    def __init__(self):
        self.pd = [
            RandomContrast(),  # RGB
            ConvertColor(current="RGB", transform='HSV'),  # HSV
            RandomSaturation(),  # HSV
            RandomHue(),  # HSV
            ConvertColor(current='HSV', transform='RGB'),  # RGB
            RandomContrast()  # RGB
        ]
        self.rand_brightness = RandomBrightness()
        self.rand_light_noise = RandomLightingNoise()

    def __call__(self, image, boxes, labels, landmarks):
        im = image.copy()
        im, boxes, labels, landmarks = self.rand_brightness(im, boxes, labels, landmarks)
        if random.randint(2):
            distort = Compose(self.pd[:-1])
        else:
            distort = Compose(self.pd[1:])
        im, boxes, labels, landmarks = distort(im, boxes, labels, landmarks)
        return self.rand_light_noise(im, boxes, labels, landmarks)


if __name__ == '__main__':
    img_transform = Compose([ConvertFromInts(),
                             PhotometricDistort(),
                             RandomSampleCrop(),
                             RandomMirror(),
                             ToPercentCoords(),
                             Resize([320, 240]),
                             SubtractMeans(np.array([127, 127, 127])),
                             DivStd(128.0),
                             ChannelFirst()])
    img_path = "../../../data/wider_face_sample/JPEGImages/0--Parade_0_Parade_marchingband_1_5.jpg"
    img = cv2.imread(img_path)
    boxes = np.array([[495, 177, 532, 228],
                      [221, 226, 259, 268]], dtype=np.float32)
    labels = np.array([1, 1], dtype=np.int64)
    landmarks = np.array([[508.357, 197.821, 526.036, 197.821, 518.964, 203.607, 510.607, 214.214, 526.036, 213.571],
                          [232.036, 237.964, 250.25, 237.964, 241.946, 245.196, 234.714, 255.107, 248.375, 255.375]],
                         dtype=np.float32)
    # (683, 1024, 3) (2, 4) (2,) (2, 10)
    print(img.shape, boxes.shape, labels.shape, landmarks.shape)
    img, boxes, labels, landmarks = img_transform(img, boxes, labels, landmarks)
    print(img.shape, boxes.shape, labels.shape, landmarks.shape)

    from ssd_data import MatchPrior
    from generate_prior import define_img_size
    priors = define_img_size(320)
    target_transform = MatchPrior(priors, 0.1, 0.2, 0.35)
    boxes, labels, landmarks = target_transform(boxes, labels, landmarks)
    # (3, 240, 320) (4420, 4) (4420,) (4420, 10)
    print(img.shape, boxes.shape, labels.shape, landmarks.shape)

