import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import sys
sys.path.append('./')
from isnet_pro.Inference2 import mask_generate
img_mode = "白色背景\\white_background"
# img_mode = "透明背景\\alpha_channel"
# dataset_path = r'D:\Doctoral_Career\Little_interest\novelAI\SD_img2img_Video\test\course1\test1'
dataset_path = r'D:\Doctoral_Career\Little_interest\novelAI\SD_img2img_Video\frame_input\048\0001.png'
background_path = ''
result_path = r'D:\Doctoral_Career\Little_interest\novelAI\SD_img2img_Video\test\test_v2f\testoutput'
ui_set_aim_bacground_rgb = "255,255,255"
IS_recstrth = 200
IS_recstrth_low= 200
# pic_generation2(img_mode,dataset_path,background_path,result_path,ui_set_aim_bacground_rgb,IS_recstrth)
img = mask_generate(img_mode,dataset_path,background_path,ui_set_aim_bacground_rgb,IS_recstrth,IS_recstrth_low,reverse_flag = False)

# mask = np.tile(mask, (1, 1, 3))
# img = Image.fromarray((mask * 255).astype(np.uint8), mode='RGB')
# img2 = Image.new("RGB", (100, 200))
# # 将图像调整大小为与 latent_mask 相同
# img = img.resize((latent_mask.shape[1], latent_mask.shape[0]))
# # 将值为 1 的像素设置为白色，值为 0 的像素设置为黑色
# img = np.array(img)
# img[latent_mask == 1] = 255
# img[latent_mask == 0] = 0

# 显示生成的图像
img.save('output.png')

