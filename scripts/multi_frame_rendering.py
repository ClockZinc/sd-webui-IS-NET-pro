# Original Xanthius (https://xanthius.itch.io/multi-frame-rendering-for-stablediffusion)
# Modified OedoSoldier [大江户战士] (https://space.bilibili.com/55123)
# Modified ClockZinc [星瞳毒唯](https://space.bilibili.com/113557956)
import numpy as np
from tqdm import trange
from PIL import Image, ImageSequence, ImageDraw, ImageFilter, PngImagePlugin

import modules.scripts as scripts
import gradio as gr

from modules import processing, shared, sd_samplers, images
from modules.processing import Processed
from modules.sd_samplers import samplers
from modules.shared import opts, cmd_opts, state
from modules import deepbooru
from modules.script_callbacks import ImageSaveParams, before_image_saved_callback
from modules.shared import opts, cmd_opts, state
from modules.sd_hijack import model_hijack

import pandas as pd

import piexif
import piexif.helper

import os
import re
import pandas as pd
MY_GLOBAL_VALUE_ITERATION_NUM = 0

def modify_global_variable(num):
    global MY_GLOBAL_VALUE_ITERATION_NUM
    MY_GLOBAL_VALUE_ITERATION_NUM = num

def gr_show(visible=True):
    return {"visible": visible, "__type__": "update"}


def gr_show_value_none(visible=True):
    return {"value": None, "visible": visible, "__type__": "update"}


def gr_show_and_load(value=None, visible=True):
    if value:
        if value.orig_name.endswith('.csv'):
            value = pd.read_csv(value.name)
        else:
            value = pd.read_excel(value.name)
    else:
        visible = False
    return {"value": value, "visible": visible, "__type__": "update"}


def sort_images(lst):
    pattern = re.compile(r"\d+(?=\.)(?!.*\d)")
    return sorted(lst, key=lambda x: int(re.search(pattern, x).group()))

class Script(scripts.Script):
    def title(self):
        return "(ISNET) Multi-frame rendering"

    def show(self, is_img2img):
        return is_img2img

    def ui(self, is_img2img):
        with gr.Row():
            input_dir = gr.Textbox(label='Input directory', lines=1)
            output_dir = gr.Textbox(label='Output directory', lines=1)
        # reference_imgs = gr.UploadButton(label="Upload Guide Frames", file_types = ['.png','.jpg','.jpeg'], live=True, file_count = "multiple")
        first_denoise = gr.Slider(
            minimum=0,
            maximum=1,
            step=0.05,
            label='Initial denoising strength',
            value=1,
            elem_id=self.elem_id("first_denoise"))
        append_interrogation = gr.Dropdown(
            label="Append interrogated prompt at each iteration", choices=[
                "None", "CLIP", "DeepBooru"], value="None")
        third_frame_image = gr.Dropdown(
            label="Third column (reference) image",
            choices=[
                "None",
                "FirstGen",
                "OriginalImg",
                "Historical",
                "SecondImg"],
            value="FirstGen")
        color_correction_enabled = gr.Checkbox(
            label="Enable color correction",
            value=False,
            elem_id=self.elem_id("color_correction_enabled"))
        unfreeze_seed = gr.Checkbox(
            label="Unfreeze seed",
            value=False,
            elem_id=self.elem_id("unfreeze_seed"))
        loopback_source = gr.Dropdown(
            label="Loopback source",
            choices=[
                "None",
                "Previous",
                "Current",
                "First"],
            value="Current")
        org_alpha = gr.Slider(
            minimum=0,
            maximum=1,
            step=0.05,
            label='图片可视度',
            value=1)
        single_mode_checkbox = gr.Checkbox(label="单图模式\\single mode")
        with gr.Row():
            use_txt = gr.Checkbox(label='Read tags from text files')

        with gr.Row():
            txt_path = gr.Textbox(
                label='Text files directory (Optional, will load from input dir if not specified)',
                lines=1)

        with gr.Row():
            use_csv = gr.Checkbox(label='Read tabular commands')
            csv_path = gr.File(
                label='.csv or .xlsx',
                file_types=['file'],
                visible=False)

        with gr.Row():
            with gr.Column():
                table_content = gr.Dataframe(visible=False, wrap=True)

        use_csv.change(
            fn=lambda x: [gr_show_value_none(x), gr_show_value_none(False)],
            inputs=[use_csv],
            outputs=[csv_path, table_content],
        )
        csv_path.change(
            fn=lambda x: gr_show_and_load(x),
            inputs=[csv_path],
            outputs=[table_content],
        )

        return [
            append_interrogation,
            input_dir,
            output_dir,
            first_denoise,
            third_frame_image,
            color_correction_enabled,
            unfreeze_seed,
            loopback_source,
            use_csv,
            table_content,
            use_txt,
            txt_path,
            single_mode_checkbox,
            org_alpha]

    def run(
            self,
            p,
            append_interrogation,
            input_dir,
            output_dir,
            first_denoise,
            third_frame_image,
            color_correction_enabled,
            unfreeze_seed,
            loopback_source,
            use_csv,
            table_content,
            use_txt,
            txt_path,
            single_mode_checkbox,
            org_alpha):
        freeze_seed = not unfreeze_seed
        # second_flag = False

        if use_csv:
            prompt_list = [i[0] for i in table_content.values.tolist()]
            prompt_list.insert(0, prompt_list.pop())

        reference_imgs = [os.path.join(input_dir,f) for f in os.listdir(input_dir) if re.match(r'.+\.(jpg|png)$',f)]
        reference_imgs = sort_images(reference_imgs)
        print(f'Will process following files: {", ".join(reference_imgs)}')

        if use_txt:
            if txt_path == "":
                files = [re.sub(r'\.(jpg|png)$', '.txt', path)
                         for path in reference_imgs]
            else:
                files = [
                    os.path.join(
                        txt_path,
                        os.path.basename(
                            re.sub(
                                r'\.(jpg|png)$',
                                '.txt',
                                path))) for path in reference_imgs]
            prompt_list = [open(file, 'r').read().rstrip('\n')
                           for file in files]

        loops = len(reference_imgs)

        processing.fix_seed(p)
        batch_count = p.n_iter

        p.batch_size = 1
        p.n_iter = 1

        output_images, info = None, None
        initial_seed = None
        initial_info = None

        initial_width = p.width
        initial_img = reference_imgs[0]  # p.init_images[0]
        p.init_images = [
            Image.open(initial_img).convert("RGB").resize(
                (initial_width, p.height), Image.ANTIALIAS)]

        # grids = []
        # all_images = []
        # original_init_image = p.init_images
        original_prompt = p.prompt
        if original_prompt != "":
            original_prompt = original_prompt.rstrip(
                ', ') + ', ' if not original_prompt.rstrip().endswith(',') else original_prompt.rstrip() + ' '
        original_denoise = p.denoising_strength
        state.job_count = loops * batch_count

        initial_color_corrections = [
            processing.setup_color_correction(
                p.init_images[0])]

        # for n in range(batch_count):
        history = None
        # frames = []
        third_image = None
        third_image_index = 0
        frame_color_correction = None

        # Reset to original init image at the start of each batch
        p.width = initial_width
        p.mask_blur = 0
        p.control_net_resize_mode = "Just Resize"
        # p.control_net_resize_mode = "Scale to Fit (Inner Fit)"
        if single_mode_checkbox :
            print("ISNET::MFR::Single mode OPEN!!!")
            third_frame_image = "None"
        for i in range(loops):
            if state.interrupted:
                break
            filename = os.path.basename(reference_imgs[i])
            p.n_iter = 1
            p.batch_size = 1
            p.do_not_save_grid = True
            # 选择controlnet输入的图像
            p.control_net_input_image = Image.open(
                reference_imgs[i]).convert("RGB").resize(
                (initial_width, p.height), Image.ANTIALIAS)

            if(i > 0):
                loopback_image = p.init_images[0]
                if loopback_source == "Current":
                    loopback_image = p.control_net_input_image
                elif loopback_source == "First":
                    loopback_image = history
                elif loopback_source == "None":
                    img2 = Image.new("RGBA", (initial_width, p.height), "white")
                    loopback_image = Image.new("RGBA", (initial_width, p.height),"white")
                    loopback_image = Image.blend(img2, loopback_image, org_alpha)

                if third_frame_image != "None":
                    p.width = initial_width * 3
                    img = Image.new("RGB", (initial_width * 3, p.height))
                    img.paste(p.init_images[0], (0, 0))
                    # img.paste(p.init_images[0], (initial_width, 0))
                    img.paste(loopback_image, (initial_width, 0))
                    if i == 1:
                        third_image = p.init_images[0]
                    img.paste(third_image, (initial_width * 2, 0))
                    p.init_images = [img]
                    if color_correction_enabled:
                        p.color_corrections = [
                            processing.setup_color_correction(img)]

                    msk = Image.new("RGB", (initial_width * 3, p.height))
                    msk.paste(Image.open(reference_imgs[i - 1]).convert("RGB").resize(
                        (initial_width, p.height), Image.ANTIALIAS), (0, 0))
                    msk.paste(p.control_net_input_image, (initial_width, 0))

                    msk.paste(Image.open(reference_imgs[third_image_index]).convert("RGB").resize(
                        (initial_width, p.height), Image.ANTIALIAS), (initial_width * 2, 0))
                    p.control_net_input_image = msk

                    latent_mask = Image.new(
                        "RGB", (initial_width * 3, p.height), "black")
                    latent_draw = ImageDraw.Draw(latent_mask)
                    latent_draw.rectangle(
                        (initial_width, 0, initial_width * 2, p.height), fill="white")
                    p.image_mask = latent_mask
                    p.denoising_strength = original_denoise
                elif not single_mode_checkbox:
                    p.width = initial_width * 2
                    img = Image.new("RGB", (initial_width * 2, p.height))
                    img.paste(p.init_images[0], (0, 0))
                    # img.paste(p.init_images[0], (initial_width, 0))
                    img.paste(loopback_image, (initial_width, 0))
                    p.init_images = [img]
                    if color_correction_enabled:
                        p.color_corrections = [
                            processing.setup_color_correction(img)]

                    msk = Image.new("RGB", (initial_width * 2, p.height))
                    msk.paste(Image.open(reference_imgs[i - 1]).convert("RGB").resize(
                        (initial_width, p.height), Image.ANTIALIAS), (0, 0))
                    msk.paste(p.control_net_input_image, (initial_width, 0))
                    p.control_net_input_image = msk
                    # frames.append(msk)

                    # latent_mask = Image.new("RGB", (initial_width*2, p.height), "white")
                    # latent_draw = ImageDraw.Draw(latent_mask)
                    # latent_draw.rectangle((0,0,initial_width,p.height), fill="black")
                    latent_mask = Image.new(
                        "RGB", (initial_width * 2, p.height), "black")
                    latent_draw = ImageDraw.Draw(latent_mask)
                    latent_draw.rectangle(
                        (initial_width, 0, initial_width * 2, p.height), fill="white")

                    # p.latent_mask = latent_mask
                    p.image_mask = latent_mask
                    p.denoising_strength = original_denoise
                elif single_mode_checkbox:
                    p.width = initial_width
                    img = loopback_image.copy()
                    p.init_images = [img]
                    if color_correction_enabled:
                        p.color_corrections = [
                            processing.setup_color_correction(img)]
                    p.control_net_input_image = p.control_net_input_image.resize(
                    (initial_width, p.height))
                    # frames.append(msk)

                    # latent_mask = Image.new("RGB", (initial_width*2, p.height), "white")
                    # latent_draw = ImageDraw.Draw(latent_mask)
                    # latent_draw.rectangle((0,0,initial_width,p.height), fill="black")
                    latent_mask = Image.new(
                        "RGB", (initial_width, p.height), "white")
                    
                    latent_draw = ImageDraw.Draw(latent_mask)
                    latent_draw.rectangle(
                        (initial_width, 0, initial_width * 2, p.height), fill="white")

                    # p.latent_mask = latent_mask
                    p.image_mask = latent_mask
                    p.denoising_strength = original_denoise
            else:
                latent_mask = Image.new(
                    "RGB", (initial_width, p.height), "white")
                # p.latent_mask = latent_mask
                p.image_mask = latent_mask
                p.denoising_strength = first_denoise
                p.control_net_input_image = p.control_net_input_image.resize(
                    (initial_width, p.height))
                init_image_for_0 = Image.open(reference_imgs[0]).convert("RGBA").resize((p.width, p.height), Image.ANTIALIAS)
                img2 = Image.new("RGBA", (initial_width, p.height), "white")
                init_image_for_0 = Image.blend(img2, init_image_for_0, org_alpha)
                p.init_images = [init_image_for_0
                    ]
                # frames.append(p.control_net_input_image)

            # if opts.img2img_color_correction:
            #     p.color_corrections = initial_color_corrections

            if append_interrogation != "None":
                p.prompt = original_prompt
                if append_interrogation == "CLIP":
                    p.prompt += shared.interrogator.interrogate(
                        p.init_images[0])
                elif append_interrogation == "DeepBooru":
                    p.prompt += deepbooru.model.tag(p.init_images[0])

            if use_csv or use_txt:
                p.prompt = original_prompt + prompt_list[i]

            # state.job = f"Iteration {i + 1}/{loops}, batch {n + 1}/{batch_count}"

            processed = processing.process_images(p)

            if initial_seed is None:
                initial_seed = processed.seed
                initial_info = processed.info

            init_img = processed.images[0]
            if(i > 0) and (not single_mode_checkbox):
                init_img = init_img.crop(
                    (initial_width, 0, initial_width * 2, p.height))
            elif single_mode_checkbox:
                init_img = init_img.crop(
                    (0, 0, initial_width, p.height))


            comments = {}
            if len(model_hijack.comments) > 0:
                for comment in model_hijack.comments:
                    comments[comment] = 1

            info = processing.create_infotext(
                p,
                p.all_prompts,
                p.all_seeds,
                p.all_subseeds,
                comments,
                0,
                0)
            pnginfo = {}
            if info is not None:
                pnginfo['parameters'] = info

            params = ImageSaveParams(init_img, p, filename, pnginfo)
            before_image_saved_callback(params)
            fullfn_without_extension, extension = os.path.splitext(
                filename)

            info = params.pnginfo.get('parameters', None)

            def exif_bytes():
                return piexif.dump({
                    'Exif': {
                        piexif.ExifIFD.UserComment: piexif.helper.UserComment.dump(info or '', encoding='unicode')
                    },
                })

            if extension.lower() == '.png':
                pnginfo_data = PngImagePlugin.PngInfo()
                for k, v in params.pnginfo.items():
                    pnginfo_data.add_text(k, str(v))

                init_img.save(
                    os.path.join(
                        output_dir,
                        filename),
                    pnginfo=pnginfo_data)

            elif extension.lower() in ('.jpg', '.jpeg', '.webp'):
                init_img.save(os.path.join(output_dir, filename))

                if opts.enable_pnginfo and info is not None:
                    piexif.insert(
                        exif_bytes(), os.path.join(
                            output_dir, filename))
            else:
                init_img.save(os.path.join(output_dir, filename))

            if third_frame_image != "None":
                if third_frame_image == "FirstGen" and i == 0:
                    third_image = init_img
                    third_image_index = 0
                elif third_frame_image == "OriginalImg" and i == 0:
                    third_image = initial_img[0]
                    third_image_index = 0
                elif third_frame_image == "Historical":
                    third_image = processed.images[0].crop(
                        (0, 0, initial_width, p.height))
                    third_image_index = (i - 1)
                elif third_frame_image == "SecondImg" and i < 3:
                    # print('\n',f'loop_length : {loops}',f'third_image_index : {third_image_index}','\n')
                    third_image = processed.images[0].crop(
                        (0, 0, initial_width, p.height))
                    third_image_index = (i - 1)
                    # print('\n',f'loop_length : {loops}',f'third_image_index : {third_image_index}','\n')



            p.init_images = [init_img]
            if(freeze_seed):
                p.seed = processed.seed
            else:
                p.seed = processed.seed + 1
            # p.seed = processed.seed
            if i == 0:
                history = init_img
            # history.append(processed.images[0])
            # frames.append(processed.images[0])

        # grid = images.image_grid(history, rows=1)
        # if opts.grid_save:
        #     images.save_image(grid, p.outpath_grids, "grid", initial_seed, p.prompt, opts.grid_format, info=info, short_filename=not opts.grid_extended_filename, grid=True, p=p)

        # grids.append(grid)
        # # all_images += history + frames
        # all_images += history

        # p.seed = p.seed+1

        # if opts.return_grid:
        #     all_images = grids + all_images

        processed = Processed(p, [], initial_seed, initial_info)

        return processed
