import warnings
warnings.filterwarnings("ignore")
# 主要有一个threshold
import os
import time
import numpy as np
from skimage import io,color
import time
from glob import glob
from tqdm import tqdm
import cv2
import torch, gc
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
from torchvision.transforms.functional import normalize
import sys
import gradio as gr
from PIL import Image
sys.path.append('./')
# sys.path.append('demo_datasets\your_dataset')
from models_DIS import *
import urllib.request
from modules.shared import state
def ui_invert_image(input_image_path,output_image_path):
    if not os.path.exists(output_image_path):
        os.makedirs(output_image_path)
    im_list = [file for ext in ['jpg', 'jpeg', 'png', 'bmp', 'tiff'] for file in glob(input_image_path + '/*.' + ext.lower())]
    for i, im_path in tqdm(enumerate(im_list), total=len(im_list)):
        input_image = cv2.imread(im_path)
        neg_img = invert_image(input_image)
        output_file = os.path.join(output_image_path, os.path.basename(im_path))
        cv2.imwrite(output_file, neg_img)

def transparent_image2whitebackground_image(img):
    alpha = img[:, :, 3]/255.0  # 获取透明度通道
    aim_bacground_rgb = (255,255,255)
    img_bacground1 = np.zeros((*img.shape[0:2], 3), dtype=np.uint8)
    img_bacground1[:] = aim_bacground_rgb
    img_bacground = img[:, :, :3] * alpha[:,:,np.newaxis] + img_bacground1 *(1- alpha[:,:,np.newaxis])
    return img_bacground

def invert_image(image):
    """
    对输入的灰度图或彩色图像进行反色处理，并返回反色图像。
    """
    if len(image.shape) == 2:  # 灰度图像
        neg_img = 255 - image
    elif len(image.shape) == 3:  # 彩色图像
        # 分离RGB通道
        r, g, b = cv2.split(image)

        # 对每个通道进行反色处理
        neg_r = 255 - r
        neg_g = 255 - g
        neg_b = 255 - b

        # 合并RGB通道
        neg_img = cv2.merge((neg_r, neg_g, neg_b))
    else:
        raise ValueError("Invalid image shape: {}".format(image.shape))
    return neg_img
# def pic_feature_abstract(target_img,normalized_gray,mode,img_bacground):
#     if mode == 'alpha_channel':
#         # 四通道生成图片
#         output_img = np.dstack((target_img, normalized_gray*255))
#         return  output_img
#     elif mode == 'white_background':
#         target_img = target_img * normalized_gray + img_bacground * (1-normalized_gray)
#         return target_img
#     elif mode == 'Solid_Color_Background':
#         target_img = target_img * normalized_gray + img_bacground * (1-normalized_gray)
#         return target_img
#     elif mode == 'self_design_Background':
#         target_img = target_img * normalized_gray + img_bacground * (1-normalized_gray)
#         return target_img

def download_model(model_name, url, model_dir='saved_models'):
    """
    下载指定名称的模型文件，并保存到指定路径下的 saved_models 文件夹中。
    如果本地已存在同名的模型文件，则不进行下载操作。

    Args:
        model_name (str): 模型文件的名称，例如 'isnet.pth'。
        url (str): 模型文件所在的 URL。
        model_dir (str): 模型文件保存的路径，默认为 'saved_models'。

    Returns:
        None
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, model_dir)
    if not os.path.exists(model_path):
        file_path_saved_models = os.path.join(current_dir, "../saved_models")
        if not os.path.exists(file_path_saved_models):
            os.mkdir(file_path_saved_models)
        os.mkdir(model_path)
    
    file_path = os.path.join(model_path, model_name)

    if not os.path.exists(file_path):
        print(f'Downloading {model_name} ......')
        with tqdm(unit='B', unit_scale=True, miniters=1, desc=model_name) as t:
            urllib.request.urlretrieve(url, filename=file_path, reporthook=lambda x, y, z: t.update(y))
        print('Download isnet.pth completed.')
# download_model(model_name = 'isnet.pth', url = 'https://huggingface.co/ClockZinc/IS-NET_pth/blob/main/isnet.pth',model_dir='..\saved_models')
download_model(model_name = 'isnet-general-use.pth', url = 'https://huggingface.co/ClockZinc/IS-NET_pth/resolve/main/isnet-general-use.pth',model_dir='../saved_models/IS-Net')

def pic_feature_abstract(target_img, normalized_gray, mode, img_bacground,IS_recstrth,IS_recstrth_low = 0.95):
    if mode == 'self_design_Background' or mode ==  'fixed_background':
        if img_bacground.shape[2]==4:
            img_bacground=transparent_image2whitebackground_image(img_bacground)
    mode_dict = {
        'alpha_channel': lambda: np.dstack((target_img, normalized_gray*255)),
        'white_background': lambda: target_img * normalized_gray + img_bacground * (1-normalized_gray),
        'Solid_Color_Background': lambda: target_img * normalized_gray + img_bacground * (1-normalized_gray),
        'self_design_Background': lambda: target_img * normalized_gray + img_bacground * (1-normalized_gray),
        'fixed_background': lambda: target_img * normalized_gray + img_bacground * (1-normalized_gray)
    }
    return mode_dict[mode]()

def pic_generation2(img_mode,dataset_path,background_path,result_path,ui_set_aim_bacground_rgb,IS_recstrth,IS_recstrth_low,reverse_flag = False):
    state.interrupted = False
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    options = {
    "透明背景\\alpha_channel": "alpha_channel",
    "白色背景\\white_background": "white_background",
    "纯色背景\\Solid_Color_Background": "Solid_Color_Background",
    "自定义背景\\self_design_Background": "self_design_Background",
    "固定背景\\fixed_background": "fixed_background"
    }   
    img_mode = options[img_mode]
    ui_set_aim_bacground_rgb = tuple(map(int, ui_set_aim_bacground_rgb.split(",")))
    IS_inference(img_mode,dataset_path,background_path,result_path,ui_set_aim_bacground_rgb,IS_recstrth/255,IS_recstrth_low/255,reverse_flag)
    print("\n:) done!")
    return ":) done"



def IS_inference(img_mode,dataset_path,background_path,result_path,ui_set_aim_bacground_rgb,IS_recstrth,IS_recstrth_low,reverse_flag):
    # ui_set_aim_bacground_rgb = (255,212,32)
    # img_mode = 'self_design_Background'
    # dataset_path="D:\Doctoral_Career\Little_interest\DIS\demo_datasets\your_dataset"  #Your dataset path
    # background_path = "D:\Doctoral_Career\Little_interest\DIS\demo_datasets\your_background_dataset"
    print("\n IS-NET_pro: start generating...")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, '..', 'saved_models', 'IS-Net', 'isnet-general-use.pth')
    # result_path="D:\Doctoral_Career\Little_interest\DIS\demo_datasets\your_dataset_result"  #The folder path that you want to save the results
    input_size=[1024,1024]
    # input_size=[1024,1024]
    net=ISNetDIS()

    if torch.cuda.is_available():
        net.load_state_dict(torch.load(model_path))
        net=net.cuda()
    else:
        net.load_state_dict(torch.load(model_path,map_location="cpu"))
        print('USING CPU!!!!')
    net.eval()   
    # bc_list = glob(background_path+".jpg")+glob(background_path+".JPG")+glob(background_path+".jpeg")+glob(background_path+".JPEG")+glob(background_path+".png")+glob(background_path+".PNG")+glob(background_path+".bmp")+glob(background_path+".BMP")+glob(background_path+".tiff")+glob(background_path+".TIFF")
    # im_list = glob(dataset_path+"\*.jpg")+glob(dataset_path+"\*.JPG")+glob(dataset_path+"\*.jpeg")+glob(dataset_path+"\*.JPEG")+glob(dataset_path+"\*.png")+glob(dataset_path+"\*.PNG")+glob(dataset_path+"\*.bmp")+glob(dataset_path+"\*.BMP")+glob(dataset_path+"\*.tiff")+glob(dataset_path+"\*.TIFF")
    im_list = [file for ext in ['jpg', 'jpeg', 'png', 'bmp', 'tiff'] for file in glob(dataset_path + '/*.' + ext.lower())]
    if img_mode =='self_design_Background' or img_mode =='fixed_background':
        bc_list = [file for ext in ['jpg', 'jpeg', 'png', 'bmp', 'tiff'] for file in glob(background_path + '/*.' + ext.lower())]


    with torch.no_grad():
        for i, im_path in tqdm(enumerate(im_list), total=len(im_list)):
            if state.interrupted:
                break
            ###
            # 输入输出的处理
            im = io.imread(im_path)
            if im.shape[2] == 4:
                im=transparent_image2whitebackground_image(im)
            if len(im.shape) < 3:
                im = im[:, :, np.newaxis]
            im_shp=im.shape[0:2]
            im_tensor = torch.tensor(im, dtype=torch.float32).permute(2,0,1)
            im_tensor = F.upsample(torch.unsqueeze(im_tensor,0), input_size, mode="bilinear").type(torch.uint8)
            image = torch.divide(im_tensor,255.0)
            image = normalize(image,[0.5,0.5,0.5],[1.0,1.0,1.0])

            if torch.cuda.is_available():
                image=image.cuda()
            result=net(image)
            result=torch.squeeze(F.upsample(result[0][0],im_shp,mode='bilinear'),0)
            ma = torch.max(result)
            mi = torch.min(result)
            result = (result-mi)/(ma-mi)
            # im_name=im_path.split('\\')[-1].split('.')[0]
            filename = os.path.basename(im_path)
            im_name = os.path.splitext(filename)[0]
            # end
            ###


            # 重读图像，原先的都不知道啥样了
            # 这是一个带有RGB值的变量。
            img1 = io.imread(im_path)
            # 这是一个零到1的变量
            grey = result.permute(1,2,0).cpu().data.numpy()

            
            # 背景
            if img_mode == 'alpha_channel':
                img_bacground = 0

            elif img_mode == 'white_background':
                aim_bacground_rgb = (255,255,255)
                img_bacground = np.zeros((*img1.shape[0:2], 3), dtype=np.uint8)
                img_bacground[:] = aim_bacground_rgb

            elif img_mode == 'Solid_Color_Background':
                aim_bacground_rgb = ui_set_aim_bacground_rgb
                img_bacground = np.zeros((*img1.shape[0:2], 3), dtype=np.uint8)
                img_bacground[:] = aim_bacground_rgb

            elif img_mode == 'self_design_Background':
                bc_path = bc_list[i]
                img_bacground = io.imread(bc_path)

            elif img_mode == 'fixed_background':
                if i==0 :
                    bc_path = bc_list[i]
                    img_bacground = io.imread(bc_path)
            grey[grey <= IS_recstrth] = 0
            grey[grey > IS_recstrth_low] = 1
            if reverse_flag:
                grey = 1 - grey
            if img1.shape[2] == 4:
                img1=transparent_image2whitebackground_image(img1)
            res_pic = pic_feature_abstract(img1,grey,mode = img_mode,img_bacground = img_bacground,IS_recstrth = IS_recstrth,IS_recstrth_low = IS_recstrth_low)
            

            io.imsave(os.path.join(result_path,im_name+".png"),np.uint8(res_pic))
            # return gr.update(value=result, visible=True, interactive=False)
            # io.imsave(os.path.join(result_path,im_name+".png"),(pic1*255).astype(np.uint8))
    del net
    torch.cuda.empty_cache()

def pic_generation_single(img_mode,dataset_path,background_path,result_path,ui_set_aim_bacground_rgb,IS_recstrth,IS_recstrth_low,reverse_flag):
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    options = {
    "透明背景\\alpha_channel": "alpha_channel",
    "白色背景\\white_background": "white_background",
    "纯色背景\\Solid_Color_Background": "Solid_Color_Background",
    "自定义背景\\self_design_Background": "self_design_Background",
    "固定背景\\fixed_background": "fixed_background"
    }   
    img_mode = options[img_mode]
    ui_set_aim_bacground_rgb = tuple(map(int, ui_set_aim_bacground_rgb.split(",")))
    res_pic,mask= IS_inference_single(img_mode,dataset_path,background_path,result_path,ui_set_aim_bacground_rgb,IS_recstrth/255,IS_recstrth_low/255,reverse_flag)
    mask = Image.fromarray(mask, mode='RGB')
    res_pic = Image.fromarray((res_pic).astype(np.uint8), mode='RGB')
    print("\n:) done!")
    return [res_pic,mask]


def IS_inference_single(img_mode,dataset_path,background_path,result_path,ui_set_aim_bacground_rgb,IS_recstrth,IS_recstrth_low,reverse_flag):
    
    # ui_set_aim_bacground_rgb = (255,212,32)
    # img_mode = 'self_design_Background'
    # dataset_path="D:\Doctoral_Career\Little_interest\DIS\demo_datasets\your_dataset"  #Your dataset path
    # background_path = "D:\Doctoral_Career\Little_interest\DIS\demo_datasets\your_background_dataset"
    print("\n IS-NET_pro: start generating...")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, '..', 'saved_models', 'IS-Net', 'isnet-general-use.pth')
    # result_path="D:\Doctoral_Career\Little_interest\DIS\demo_datasets\your_dataset_result"  #The folder path that you want to save the results
    input_size=[1024,1024]
    # input_size=[1024,1024]
    net=ISNetDIS()

    if torch.cuda.is_available():
        net.load_state_dict(torch.load(model_path))
        net=net.cuda()
    else:
        net.load_state_dict(torch.load(model_path,map_location="cpu"))
        print('USING CPU!!!!')
    net.eval()   
    # bc_list = glob(background_path+".jpg")+glob(background_path+".JPG")+glob(background_path+".jpeg")+glob(background_path+".JPEG")+glob(background_path+".png")+glob(background_path+".PNG")+glob(background_path+".bmp")+glob(background_path+".BMP")+glob(background_path+".tiff")+glob(background_path+".TIFF")
    # im_list = glob(dataset_path+"\*.jpg")+glob(dataset_path+"\*.JPG")+glob(dataset_path+"\*.jpeg")+glob(dataset_path+"\*.JPEG")+glob(dataset_path+"\*.png")+glob(dataset_path+"\*.PNG")+glob(dataset_path+"\*.bmp")+glob(dataset_path+"\*.BMP")+glob(dataset_path+"\*.tiff")+glob(dataset_path+"\*.TIFF")
    im_list = [dataset_path]
    # im_list = [file for ext in ['jpg', 'jpeg', 'png', 'bmp', 'tiff'] for file in glob(dataset_path + '/*.' + ext.lower())]
    # print(im_list)
    if img_mode =='self_design_Background' or img_mode =='fixed_background':
        bc_list = background_path


    with torch.no_grad():
        for i, im_path in tqdm(enumerate(im_list), total=len(im_list)):
            ###
            # 输入输出的处理
            im = io.imread(im_path)
            if im.shape[2] == 4:
                im=transparent_image2whitebackground_image(im)
            if len(im.shape) < 3:
                im = im[:, :, np.newaxis]
            im_shp=im.shape[0:2]
            im_tensor = torch.tensor(im, dtype=torch.float32).permute(2,0,1)
            im_tensor = F.upsample(torch.unsqueeze(im_tensor,0), input_size, mode="bilinear").type(torch.uint8)
            image = torch.divide(im_tensor,255.0)
            image = normalize(image,[0.5,0.5,0.5],[1.0,1.0,1.0])

            if torch.cuda.is_available():
                image=image.cuda()
            result=net(image)
            result=torch.squeeze(F.upsample(result[0][0],im_shp,mode='bilinear'),0)
            ma = torch.max(result)
            mi = torch.min(result)
            result = (result-mi)/(ma-mi)
            # im_name=im_path.split('\\')[-1].split('.')[0]
            filename = os.path.basename(im_path)
            im_name = os.path.splitext(filename)[0]
            # end
            ###


            # 重读图像，原先的都不知道啥样了
            # 这是一个带有RGB值的变量。
            img1 = io.imread(im_path)
            # 这是一个零到1的变量
            grey = result.permute(1,2,0).cpu().data.numpy()

            
            # 背景
            if img_mode == 'alpha_channel':
                img_bacground = 0

            elif img_mode == 'white_background':
                aim_bacground_rgb = (255,255,255)
                img_bacground = np.zeros((*img1.shape[0:2], 3), dtype=np.uint8)
                img_bacground[:] = aim_bacground_rgb

            elif img_mode == 'Solid_Color_Background':
                aim_bacground_rgb = ui_set_aim_bacground_rgb
                img_bacground = np.zeros((*img1.shape[0:2], 3), dtype=np.uint8)
                img_bacground[:] = aim_bacground_rgb

            elif img_mode == 'self_design_Background':
                bc_path = bc_list[i]
                img_bacground = io.imread(bc_path)

            elif img_mode == 'fixed_background':
                if i==0 :
                    bc_path = bc_list[i]
                    img_bacground = io.imread(bc_path)

            if img1.shape[2] == 4:
                img1=transparent_image2whitebackground_image(img1)
            grey[grey <= IS_recstrth] = 0
            grey[grey > IS_recstrth_low] = 1
            if reverse_flag:
                grey = 1 - grey
            res_pic = pic_feature_abstract(img1,grey,mode = img_mode,img_bacground = img_bacground,IS_recstrth = IS_recstrth, IS_recstrth_low = IS_recstrth_low)
            
            res_pic = np.uint8(res_pic)

            grey = np.tile(grey, (1, 1, 3))
            grey = (grey*255).astype(np.uint8)
            io.imsave(os.path.join(result_path,im_name+".png"),res_pic)
            io.imsave(os.path.join(result_path,"mask"+im_name+".png"),grey)
            # return gr.update(value=result, visible=True, interactive=False)
            # io.imsave(os.path.join(result_path,im_name+".png"),(pic1*255).astype(np.uint8))

            return res_pic,grey
    del net
    torch.cuda.empty_cache()

def mask_generate(img_mode,dataset_path,output_dir,ui_set_aim_bacground_rgb,IS_recstrth,IS_recstrth_low,reverse_flag):
    """
    Output is purely mask
    """
    options = {
    "透明背景\\alpha_channel": "alpha_channel",
    "白色背景\\white_background": "white_background",
    "纯色背景\\Solid_Color_Background": "Solid_Color_Background",
    "自定义背景\\self_design_Background": "self_design_Background",
    "固定背景\\fixed_background": "fixed_background"
    }   
    img_mode = options[img_mode]
    ui_set_aim_bacground_rgb = tuple(map(int, ui_set_aim_bacground_rgb.split(",")))
    mask = IS_inference_mask(img_mode,dataset_path,output_dir,ui_set_aim_bacground_rgb,IS_recstrth/255,IS_recstrth_low/255,reverse_flag)
    print("\n:) mask generate done!")
    return mask
def IS_inference_mask(img_mode,dataset_path,output_dir,ui_set_aim_bacground_rgb,IS_recstrth,IS_recstrth_low,reverse_flag):
    
    # ui_set_aim_bacground_rgb = (255,212,32)
    # img_mode = 'self_design_Background'
    # dataset_path="D:\Doctoral_Career\Little_interest\DIS\demo_datasets\your_dataset"  #Your dataset path
    # background_path = "D:\Doctoral_Career\Little_interest\DIS\demo_datasets\your_background_dataset"
    print("\n IS-NET_pro: start generating...")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, '..', 'saved_models', 'IS-Net', 'isnet-general-use.pth')
    # result_path="D:\Doctoral_Career\Little_interest\DIS\demo_datasets\your_dataset_result"  #The folder path that you want to save the results
    input_size=[1024,1024]
    # input_size=[1024,1024]
    net=ISNetDIS()

    if torch.cuda.is_available():
        net.load_state_dict(torch.load(model_path))
        net=net.cuda()
    else:
        net.load_state_dict(torch.load(model_path,map_location="cpu"))
        print('USING CPU!!!!')
    net.eval()   
    # bc_list = glob(background_path+".jpg")+glob(background_path+".JPG")+glob(background_path+".jpeg")+glob(background_path+".JPEG")+glob(background_path+".png")+glob(background_path+".PNG")+glob(background_path+".bmp")+glob(background_path+".BMP")+glob(background_path+".tiff")+glob(background_path+".TIFF")
    # im_list = glob(dataset_path+"\*.jpg")+glob(dataset_path+"\*.JPG")+glob(dataset_path+"\*.jpeg")+glob(dataset_path+"\*.JPEG")+glob(dataset_path+"\*.png")+glob(dataset_path+"\*.PNG")+glob(dataset_path+"\*.bmp")+glob(dataset_path+"\*.BMP")+glob(dataset_path+"\*.tiff")+glob(dataset_path+"\*.TIFF")
    # im_list = [dataset_path]
    im_list = [file for ext in ['jpg', 'jpeg', 'png', 'bmp', 'tiff'] for file in glob(dataset_path + '/*.' + ext.lower())]
    # print(im_list)
    if img_mode =='self_design_Background' or img_mode =='fixed_background':
        bc_list = background_path


    with torch.no_grad():
        for i, im_path in tqdm(enumerate(im_list), total=len(im_list)):
            if state.interrupted:
                break
            ###
            # 输入输出的处理
            im = io.imread(im_path)
            if im.shape[2] == 4:
                im=transparent_image2whitebackground_image(im)
            if len(im.shape) < 3:
                im = im[:, :, np.newaxis]
            im_shp=im.shape[0:2]
            im_tensor = torch.tensor(im, dtype=torch.float32).permute(2,0,1)
            im_tensor = F.upsample(torch.unsqueeze(im_tensor,0), input_size, mode="bilinear").type(torch.uint8)
            image = torch.divide(im_tensor,255.0)
            image = normalize(image,[0.5,0.5,0.5],[1.0,1.0,1.0])

            if torch.cuda.is_available():
                image=image.cuda()
            result=net(image)
            result=torch.squeeze(F.upsample(result[0][0],im_shp,mode='bilinear'),0)
            ma = torch.max(result)
            mi = torch.min(result)
            result = (result-mi)/(ma-mi)
            # im_name=im_path.split('\\')[-1].split('.')[0]
            filename = os.path.basename(im_path)
            im_name = os.path.splitext(filename)[0]
            # end
            ###


            # 重读图像，原先的都不知道啥样了
            # 这是一个带有RGB值的变量。
            img1 = io.imread(im_path)
            # 这是一个零到1的变量
            grey = result.permute(1,2,0).cpu().data.numpy()

            
            # 背景
            if img_mode == 'alpha_channel':
                img_bacground = 0

            elif img_mode == 'white_background':
                aim_bacground_rgb = (255,255,255)
                img_bacground = np.zeros((*img1.shape[0:2], 3), dtype=np.uint8)
                img_bacground[:] = aim_bacground_rgb

            elif img_mode == 'Solid_Color_Background':
                aim_bacground_rgb = ui_set_aim_bacground_rgb
                img_bacground = np.zeros((*img1.shape[0:2], 3), dtype=np.uint8)
                img_bacground[:] = aim_bacground_rgb

            elif img_mode == 'self_design_Background':
                bc_path = bc_list[i]
                img_bacground = io.imread(bc_path)

            elif img_mode == 'fixed_background':
                if i==0 :
                    bc_path = bc_list[i]
                    img_bacground = io.imread(bc_path)



            grey[grey <= IS_recstrth] = 0
            grey[grey > IS_recstrth] = 1
            if reverse_flag:
                grey = 1 - grey
            grey = np.tile(grey, (1, 1, 3))
            grey = (grey * 255).astype(np.uint8)
            # return gr.update(value=result, visible=True, interactive=False)
            io.imsave(os.path.join(output_dir,im_name+".png"),grey)
    del net
    torch.cuda.empty_cache()
    return 1
if __name__ == '__main__':
    # img = io.imread(r'D:\Doctoral_Career\Little_interest\novelAI\SD_img2img_Video\test\course2\output\0002.png',)
    # print(img.shape[2])
    # if img.shape[2] == 4:
    #     img_bacground=transparent_image2whitebackground_image(img)
    # io.imsave(r'D:\Doctoral_Career\Little_interest\novelAI\SD_img2img_Video\test\course2\output\00020.png',np.uint8(img_bacground))
    img_mode = "白色背景\\white_background"
    # img_mode = "透明背景\\alpha_channel"
    # dataset_path = r'D:\Doctoral_Career\Little_interest\novelAI\SD_img2img_Video\test\course1\test1'
    dataset_path = r'D:\Doctoral_Career\Little_interest\novelAI\SD_img2img_Video\test\test_v2f\testinput\0028.png'
    background_path = ''
    result_path = r'D:\Doctoral_Career\Little_interest\novelAI\SD_img2img_Video\test\test_v2f\testoutput'
    ui_set_aim_bacground_rgb = "255,255,255"
    IS_recstrth = 10
    # pic_generation2(img_mode,dataset_path,background_path,result_path,ui_set_aim_bacground_rgb,IS_recstrth)
    pic_generation_single(img_mode,dataset_path,background_path,result_path,ui_set_aim_bacground_rgb,IS_recstrth,IS_recstrth_low = 20,reverse_flag = False)