import cv2
import os,re
from tqdm import tqdm
import numpy as np
import modules.scripts as scripts
import modules.shared as shared
from modules.processing import StableDiffusionProcessingImg2Img
import gradio as gr
def isHasNoFile(folder_path):
    file_list = os.listdir(folder_path)
    if len(file_list) == 0:
        return True
    else:
        return False
    
class Script(scripts.Script):
    def title(self):
        return "(batch only) batch prompt adder"

    def show(self, is_img2img):
        return is_img2img

    def ui(self, is_img2img):
        self.current_txt_count = 0 # 初始化txt文本计数器
        self.original_prompt = ""
        with gr.Column():
            input_dir = gr.Textbox(label='batch Image Input directory', lines=1)
            output_dir = gr.Textbox(label='batch Image Output directory', lines=1)
            txt_dir = gr.Textbox(label='batch Image prompt.txt Input directory', lines=1)
        return [input_dir,output_dir,txt_dir]
    
    # 直接对p进行修改就行了,不需要返回值操作
    def run(self,p:StableDiffusionProcessingImg2Img,input_dir,output_dir,txt_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 如果没有文件就直接置零
        if isHasNoFile(output_dir):
            self.current_txt_count = 0
            self.original_prompt = p.prompt

        # 获取全部文件的路径
        images = list(shared.walk_files(input_dir, allowed_extensions=(".png", ".jpg", ".jpeg", ".webp")))

        # 生成prompt_list
        if txt_dir == "":
            files = [re.sub(r'\.(jpg|png|jpeg|webp)$', '.txt', path)
                        for path in images]
        else:
            files = [
                os.path.join(
                txt_dir,
                os.path.basename(re.sub(r'\.(jpg|png|jpeg|webp)$','.txt',path))) for path in images]
            
        prompt_list = [open(file, 'r').read().rstrip('\n')
                        for file in files]
        
        
        # 增加prompt
        p.prompt = self.original_prompt + prompt_list[self.current_txt_count]

        # 测试用的
        # print(f'count :: {self.current_txt_count}\n',
        #       f'prompt :: {p.prompt}\n')
        # 指针增加
        self.current_txt_count += 1
        
        
        


