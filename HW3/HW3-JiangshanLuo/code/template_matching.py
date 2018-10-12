import numpy as np
import cv2
from os import listdir, makedirs
from os.path import isfile, join, abspath, exists
import time

def image_pyramid(img, base_ratio=1):
    rescale_ratios = [base_ratio-0.4, base_ratio-0,2, base_ratio, base_ratio+0.2, base_ratio+0.4]
    rescaled_images = []

    for ratio in rescale_ratios:
        rescaled_x = int(img.shape[0] * ratio)
        rescaled_y = int(img.shape[1] * ratio)
        if rescaled_x > 0 and rescaled_y > 0:
            rescaled_images.append(cv2.resize(img, (rescaled_y, rescaled_x)))
    rescaled_images = rescaled_images
    return rescaled_images

# for binary images
def ssd(img, template):
    ssd_value = np.sum(img != template)
    return ssd_value

# for grayscale images
def ncc(img, template):
    ncc_value = 0
    s_avg = np.mean(img)
    s_std = np.std(img)
    m_avg = np.mean(template)
    m_std = np.std(template)
    # (si - s_avg) * (mi - m_avg)
    matrix_sum = np.matmul((img.flatten() - s_avg), (template.flatten() - m_avg).T)
    # s_std * m_std
    std_mul = s_std * m_std
    # sum of coefficient correlation
    scc = matrix_sum / std_mul
    # ncc
    size = img.shape[0] * img.shape[1]
    ncc_value = scc / size
    return ncc_value

# return the img inside the bounding box
def remove_padding(binary_imgs):
    img_noPadding = []
    for img in binary_imgs:
        _, contours_opencv, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # choose the largest size object
        largest_index = 0
        largest_area = cv2.contourArea(contours_opencv[largest_index])
        for index in range(len(contours_opencv)):
            area = cv2.contourArea(contours_opencv[index])
            if area > largest_area:
                largest_area = area
                largest_index = index
        
        x, y, w, h = cv2.boundingRect(contours_opencv[largest_index])
        # print(x, y, w, h)
        img_noPadding.append(img[y:y+h,x:x+w])
        #print(img[y:y+h,x:x+w].shape)
    
    return img_noPadding

'''
    input:
        - imgs: an array of binary image with one blob inside
        - templates: an array of templates (binary image)
        - method: binary_ssd or grayscale_ncc
    return:
        - ssd_value and min_ssd for each (image, template) pair (here: 0->circle, 1->triangle,2->square)
    description:
        remove padding pixels -> resize templates to the same size as image (no bias on size) -> template matching algorithm
'''
def template_matching(imgs, templates, method='binary_ssd'):
    shape_recognition = []
    shape_value = []

    if method == 'binary_ssd':
        for each_img in imgs:
            ssd_values = []
            for template in templates:
                rescaled_template = cv2.resize(template, (each_img.shape[1], each_img.shape[0]))

                rotated_ssd = []
                rows,cols = rescaled_template.shape
                for angle in [-15, 0, 15]:
                    rotation_axis = cv2.getRotationMatrix2D((cols/2,rows/2), angle, 1)
                    rotation_temple = cv2.warpAffine(rescaled_template, rotation_axis, (cols,rows))
                    rotated_ssd.append(ssd(each_img, rotation_temple))
                ssd_values.append(np.min(rotated_ssd))

            # print('prediction:', np.argmin(ssd_values))
            # print(ssd_values)
            shape_recognition.append(np.argmin(ssd_values))
            shape_value.append(ssd_values[shape_recognition[-1]])
    elif method == 'grayscale_ncc':
        for each_img in imgs:
            ncc_values = []
            for template in templates:
                rescaled_template = cv2.resize(template, (each_img.shape[1], each_img.shape[0]))

                rotated_ncc = []
                rows,cols = rescaled_template.shape
                for angle in [-15, 0, 15]:
                    rotation_axis = cv2.getRotationMatrix2D((cols/2,rows/2), angle, 1)
                    rotation_temple = cv2.warpAffine(rescaled_template, rotation_axis, (cols,rows))
                    rotated_ncc.append(ncc(each_img, rotation_temple))
                ncc_values.append(np.max(rotated_ncc))

            # print('prediction:', np.argmax(ncc_values))
            # print(ncc_values)
            shape_recognition.append(np.argmax(ncc_values))
            shape_value.append(ncc_values[shape_recognition[-1]])

    return shape_recognition, shape_value