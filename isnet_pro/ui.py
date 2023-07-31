import gradio as gr
from isnet_pro.video2frame import video2frame,ui_frame2video
from isnet_pro.Inference2 import pic_generation_single,pic_generation2,ui_invert_image, mask_generate
import modules.shared as shared
def gr_show(visible=True):
    return {"visible": visible, "__type__": "update"}
def on_ui_tabs():
    with gr.Blocks(analytics_enabled=False) as pro_interface:
        with gr.Row().style(equal_height=False):
            with gr.Column(variant='panel'):
                with gr.Tabs():
                    with gr.TabItem(label='frame2video & video2frame'):
                        with gr.Row(variant='panel'):
                            with gr.Column(variant='panel'):
                                gr.Markdown(""" 
                                ## 视频生成'帧'\\video2frame
                                在下面上传你的视频吧\\upload your video
                                """)
                                video_input_dir = gr.Video(lable='上传视频\\upload video',source='upload',interactive=True)
                                video_input_dir.style(width=300)
                                with gr.Row(variant='panel'):
                                    aim_fps_checkbox = gr.Checkbox(label="启用输出帧率控制\\enable the ouput fps control")
                                    aim_fps = gr.Slider(
                                        minimum=1,
                                        maximum=60,
                                        step=1,
                                        label='输出帧率\\output fps',
                                        value=30,interactive=True)
                                with gr.Row(variant='panel'):
                                    time_range_checkbox = gr.Checkbox(label="启用时间段裁剪\\enable video cut")
                                    aim_start_time = gr.Number(value=0,label="裁剪起始时间(s)\\start_time",)
                                    aim_end_time = gr.Number(value=0,label="裁剪停止时间(s)\\end_time")
                                frame_output_dir = gr.Textbox(label='图片输出地址\\Frame Output directory', lines=1,placeholder='output\\folder')
                                btn = gr.Button(value="gene_frame")
                                out = gr.Textbox(label="log info",interactive=False,visible=True,placeholder="output log")
                                btn.click(video2frame, inputs=[video_input_dir, frame_output_dir,aim_fps_checkbox,aim_fps,time_range_checkbox,aim_start_time,aim_end_time],outputs=out)
                    # with gr.TabItem(label='video2frame'):
                    #     with gr.Row(variant='panel'):
                            with gr.Column(variant='panel'):
                                gr.Markdown(""" 
                                ## 帧生成'视频'\\frame2video
                                由图片转化为视频，注意这里只需要给出生成视频的地址即可，不要文件名！！！！  
                                本拓展由 [_星瞳毒唯](https://space.bilibili.com/113557956)编写  
                                本拓展GitHub项目在 [_github_sd-webui-IS-NET-pro](https://github.com/ClockZinc/sd-webui-IS-NET-pro)  
                                本拓展使用的算法有使用的DIS的开源项目 [_github_DIS](https://github.com/xuebinqin/DIS)  
                                有任何问题均可b站私信我,我看心情回答,本项目不收取任何费用！！  
                                
                                """)
                                fps = gr.Slider(
                                    minimum=1,
                                    maximum=60,
                                    step=1,
                                    label='FPS',
                                    value=30)
                                frame_input_dir = gr.Textbox(label='图片输入地址\\Frame Input directory', lines=1,placeholder='input\\folder')
                                video_output_dir = gr.Textbox(label='视频输出地址\\Video Output directory', lines=1,placeholder='output\\folder')
                                f2v_mode = gr.Dropdown(
                                    label="video out",
                                    choices=[
                                        '.mp4',
                                        '.avi',
                                        ],
                                    value='.mp4')
                                btn1 = gr.Button(value="gene_video")
                                out1 = gr.Textbox(label="log info",interactive=False,visible=True,placeholder="output log")

                                btn1.click(ui_frame2video, inputs=[frame_input_dir, video_output_dir,fps,f2v_mode],outputs=out1)
                    with gr.TabItem(label='ISNETpro2'):
                        with gr.Row(variant='panel'):
                            with gr.Column(variant='panel'):
                                gr.Markdown(""" 
                                ## 图片背景去除\\Remove Background
                                更干净的背景！！\\better performance
                                """)
                                IS_frame_input_dir = gr.Textbox(label='图片输入地址\\frame_input_dir',lines=1,placeholder='input\\folder')
                                IS_frame_output_dir = gr.Textbox(label='图片输出地址\\frame_output_dir',lines=1,placeholder='output\\folder',value='./outputs/Isnet_output')
                                IS_mode = gr.Dropdown(
                                    label="图片输出模式\\frame output mode",
                                    choices=[
                                        "透明背景\\alpha_channel",
                                        "白色背景\\white_background",
                                        "纯色背景\\Solid_Color_Background",
                                        "自定义背景\\self_design_Background",
                                        "固定背景\\fixed_background"],
                                    value="白色背景\\white_background")
                                
                                with gr.Row(variant='panel'):
                                    with gr.Column(variant='panel'):
                                        reverse_checkbox = gr.Checkbox(label="反向选取\\reverse mode")
                                        use_other_mask_checkbox = gr.Checkbox(label="使用外部的蒙版\\another mask")
                                    IS_recstrth = gr.Slider(
                                        minimum=1,
                                        maximum=255,
                                        step=1,
                                        label="背景去除强度\\background remove strength",
                                        value=30)
                                    IS_recstrth_low = gr.Slider(
                                        minimum=1,
                                        maximum=255,
                                        step=1,
                                        label="主体保留强度\\Principal retention strength",
                                        value=40)
                                IS_mask_dir = gr.Textbox(label="蒙版图片地址\\mask_input_dir",lines=1,placeholder='mask dir that generate by yourself',visible=False)
                                use_other_mask_checkbox.change(fn = gr_show, inputs=[use_other_mask_checkbox],outputs=[IS_mask_dir])

                                markdown1 = gr.Markdown(""" 
                                ### 可选信息填写\\Optional Info
                                下面两个请根据自己情况填写，纯色背景的时候需要填写目标RGB，自定义背景和固定背景需要填写背景图片地址,固定背景默认文件夹中的第一张图片  
                                When using a `Solid_Color_Background`, you need to fill in the target RGB.
                                For `self_design_background` and `fixed_background`, you need to fill in the background image address.
                                The `fixed_background` mode will uses the FIRST image in the image_dir.
                                """,visible=False)                                
                                IS_rgb_input = gr.Textbox(label="目标RGB\\target RGB",value='10,154,56',visible=False)
                                IS_bcgrd_dir = gr.Textbox(label="背景图片地址\\background_input_dir",lines=1,placeholder='input\\folder, which is NOT input\\image.png!!!',visible=False)
                                
                                def IS_mode_change(choices):
                                    if choices == "纯色背景\\Solid_Color_Background" :
                                        return gr.update(visible=True),gr.update(visible=False),gr.update(visible=True)
                                    elif choices == "自定义背景\\self_design_Background" or choices=="固定背景\\fixed_background":
                                        return gr.update(visible=False),gr.update(visible=True),gr.update(visible=True)
                                    else:
                                        return gr.update(visible=False),gr.update(visible=False),gr.update(visible=False)
                                IS_mode.change(fn = IS_mode_change,inputs=[IS_mode], outputs=[IS_rgb_input,IS_bcgrd_dir,markdown1]) 
                                with gr.Column(variant='panel'):
                                    IS_out1 = gr.Textbox(label="log info",interactive=False,visible=True,placeholder="output log")
                                    IS_btn2 = gr.Button(value="开始批量生成\\gene_batch_frame")
                                    IS_btn4 = gr.Button(value="仅生成蒙版\\gene_mask_only")
                                    IS_btn3 = gr.Button(value="中断\\Interupt")
                                    
                                IS_btn4.click(fn=mask_generate,inputs=[IS_mode,IS_frame_input_dir,IS_frame_output_dir,IS_rgb_input,IS_recstrth,IS_recstrth_low,reverse_checkbox],outputs=[])
                                IS_btn2.click(pic_generation2,inputs=[IS_mode,IS_frame_input_dir,IS_bcgrd_dir,IS_frame_output_dir,IS_rgb_input,IS_recstrth,IS_recstrth_low,reverse_checkbox,IS_mask_dir],outputs=IS_out1)
                                IS_btn3.click(fn=lambda: shared.state.interrupt(),inputs=[],outputs=[])
                            with gr.Column(variant='panel'):
                                with gr.Tabs():
                                    with gr.TabItem(label='自动抠图\\auto Image cutout'):
                                        IS_single_image_input_numpy_data = gr.Image(source='upload',type='numpy')
                                        IS_single_mode_btn = gr.Button(value="开始生成单图\\gene_single_image")
                                        image_show_path = gr.Gallery(label='output images').style(grid=2,height=720)
                                        IS_single_mode_btn.click(fn=pic_generation_single,
                                                                inputs=[IS_mode,IS_single_image_input_numpy_data,IS_bcgrd_dir,IS_frame_output_dir,IS_rgb_input,IS_recstrth,IS_recstrth_low,reverse_checkbox],
                                                                outputs=[image_show_path])
                                    with gr.TabItem(label='手动抠图\\auto Image cutout'):
                                        with gr.Row(variant='panel'):
                                            IS_single_image_input_numpy_data1 = gr.Image(source='upload',type='numpy')
                                            IS_single_mask_image_numpy_data = gr.Image(source='upload',type='numpy')
                                        IS_single_mode_btn1 = gr.Button(value="开始生成单图\\gene_single_image")
                                        image_show_path1 = gr.Gallery(label='output images').style(grid=2,height=720)
                                        IS_single_mode_btn1.click(fn=pic_generation_single,
                                                                inputs=[IS_mode,IS_single_image_input_numpy_data1,IS_bcgrd_dir,IS_frame_output_dir,IS_rgb_input,IS_recstrth,IS_recstrth_low,reverse_checkbox,IS_single_mask_image_numpy_data],
                                                                outputs=[image_show_path1])
                                                    
                    with gr.TabItem(label='InventColor'):
                        with gr.Row(variant='panel'):
                            with gr.Column(variant='panel'):
                                gr.Markdown(""" 
                                ## 图片反色\\Invert color
                                """)
                                IS_frame_input_dir = gr.Textbox(label='图片输入地址\\frame_input_dir',lines=1,placeholder='input\\folder')
                                IS_frame_output_dir = gr.Textbox(label='图片输出地址\\frame_output_dir',lines=1,placeholder='output\\folder')
                            inv_btn = gr.Button(value="开始生成\\gene_frame")
                            inv_btn.click(fn=ui_invert_image,inputs=[IS_frame_input_dir,IS_frame_output_dir])
                            
    return [(pro_interface, "isnet_Pro", "isnet_Pro")]