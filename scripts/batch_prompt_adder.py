import os,re
from contextlib import closing
from pathlib import Path

from tqdm import tqdm
import numpy as np
from PIL import Image, ImageOps, ImageFilter, ImageEnhance, ImageChops, UnidentifiedImageError

import gradio as gr

from modules import sd_samplers, images as imgutil
from modules.generation_parameters_copypaste import create_override_settings_dict, parse_generation_parameters
from modules.processing import Processed, StableDiffusionProcessingImg2Img, process_images
from modules.shared import opts, state
from modules.images import save_image
import modules.shared as shared
import modules.processing as processing
from modules.ui import plaintext_to_html
import modules.scripts as scripts

def isHasNoFile(folder_path):
    file_list = os.listdir(folder_path)
    if len(file_list) == 0:
        return True
    else:
        return False

def process_batch(p, input_dir, output_dir, inpaint_mask_dir, args, to_scale=False, scale_by=1.0, use_png_info=False, png_info_props=None, png_info_dir=None):
    processing.fix_seed(p)

    images = list(shared.walk_files(input_dir, allowed_extensions=(".png", ".jpg", ".jpeg", ".webp")))

    is_inpaint_batch = False
    if inpaint_mask_dir:
        inpaint_masks = shared.listfiles(inpaint_mask_dir)
        is_inpaint_batch = bool(inpaint_masks)

        if is_inpaint_batch:
            print(f"\nInpaint batch is enabled. {len(inpaint_masks)} masks found.")

    print(f"Will process {len(images)} images, creating {p.n_iter * p.batch_size} new images for each.")

    save_normally = output_dir == ''

    p.do_not_save_grid = True
    p.do_not_save_samples = not save_normally

    state.job_count = len(images) * p.n_iter

    # extract "default" params to use in case getting png info fails
    prompt = p.prompt
    negative_prompt = p.negative_prompt
    seed = p.seed
    cfg_scale = p.cfg_scale
    sampler_name = p.sampler_name
    steps = p.steps

    for i, image in enumerate(images):
        state.job = f"{i+1} out of {len(images)}"
        if state.skipped:
            state.skipped = False

        if state.interrupted:
            break

        try:
            img = Image.open(image)
        except UnidentifiedImageError as e:
            print(e)
            continue
        # Use the EXIF orientation of photos taken by smartphones.
        img = ImageOps.exif_transpose(img)

        if to_scale:
            p.width = int(img.width * scale_by)
            p.height = int(img.height * scale_by)

        p.init_images = [img] * p.batch_size

        image_path = Path(image)
        if is_inpaint_batch:
            # try to find corresponding mask for an image using simple filename matching
            if len(inpaint_masks) == 1:
                mask_image_path = inpaint_masks[0]
            else:
                # try to find corresponding mask for an image using simple filename matching
                mask_image_dir = Path(inpaint_mask_dir)
                masks_found = list(mask_image_dir.glob(f"{image_path.stem}.*"))

                if len(masks_found) == 0:
                    print(f"Warning: mask is not found for {image_path} in {mask_image_dir}. Skipping it.")
                    continue

                # it should contain only 1 matching mask
                # otherwise user has many masks with the same name but different extensions
                mask_image_path = masks_found[0]

            mask_image = Image.open(mask_image_path)
            p.image_mask = mask_image

        if use_png_info:
            try:
                info_img = img
                if png_info_dir:
                    info_img_path = os.path.join(png_info_dir, os.path.basename(image))
                    info_img = Image.open(info_img_path)
                geninfo, _ = imgutil.read_info_from_image(info_img)
                parsed_parameters = parse_generation_parameters(geninfo)
                parsed_parameters = {k: v for k, v in parsed_parameters.items() if k in (png_info_props or {})}
            except Exception:
                parsed_parameters = {}

            p.prompt = prompt + (" " + parsed_parameters["Prompt"] if "Prompt" in parsed_parameters else "")
            p.negative_prompt = negative_prompt + (" " + parsed_parameters["Negative prompt"] if "Negative prompt" in parsed_parameters else "")
            p.seed = int(parsed_parameters.get("Seed", seed))
            p.cfg_scale = float(parsed_parameters.get("CFG scale", cfg_scale))
            p.sampler_name = parsed_parameters.get("Sampler", sampler_name)
            p.steps = int(parsed_parameters.get("Steps", steps))


        # 接入prompt
        try:
            file = re.sub(r'\.(jpg|png|jpeg|webp)$', '.txt', image)
            current_prompt = open(file, 'r').read().rstrip('\n')
            p.prompt = prompt + current_prompt
            print(f'current ite :{i}\n',
                f'current prompt : {p.prompt}\n') 
        except:
            print("no txt detect")
        
        proc = process_images(p)

        for n, processed_image in enumerate(proc.images):
            filename = image_path.stem
            infotext = proc.infotext(p, 0)
            relpath = os.path.dirname(os.path.relpath(image, input_dir))

            if n > 0:
                filename += f"-{n}"

            if not save_normally:
                os.makedirs(os.path.join(output_dir, relpath), exist_ok=True)
                if processed_image.mode == 'RGBA':
                    processed_image = processed_image.convert("RGB")
                save_image(processed_image, os.path.join(output_dir, relpath), None, extension=opts.samples_format, info=infotext, forced_filename=filename, save_to_dirs=False)
            break
        
    return proc

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
            inpaint_mask_dir = gr.Textbox(label='inpaint Image Inpur directory', lines=1)
            # txt_dir = gr.Textbox(label='batch Image prompt.txt Input directory', lines=1)
            scale_by = gr.Slider(
            minimum=0,
            maximum=20,
            step=0.01,
            label='scale_by',
            value=1)
            # scale_by = gr.Number(label='scale_by',value=1)
        return [input_dir,output_dir,inpaint_mask_dir,scale_by]
    
    # 直接对p进行修改就行了,不需要返回值操作
    def run(self,p:StableDiffusionProcessingImg2Img,input_dir,output_dir,inpaint_mask_dir,scale_by):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        proc = process_batch(p, input_dir, output_dir, inpaint_mask_dir, p.script_args, to_scale=True, scale_by=scale_by, use_png_info=False, png_info_props=None, png_info_dir=None)
        
        return proc

        
        
        


