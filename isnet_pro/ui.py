import gradio as gr
from isnet_pro.video2frame import video2frame,ui_frame2video
from isnet_pro.Inference import pic_generation,ui_invert_image
def on_ui_tabs():
    with gr.Blocks(analytics_enabled=False) as pro_interface:
        with gr.Row().style(equal_height=False):
            with gr.Column(variant='panel'):
                with gr.Tabs():
                    with gr.TabItem(label='frame2video & video2frame'):
                        with gr.Row(variant='panel'):
                            with gr.Column(variant='panel'):
                                gr.Markdown(""" 
                                ## 视频生成'帧'/video2frame
                                由视频生成帧,注意视频地址要具体到哪一个视频,在下面上传你的视频吧  
                                """)
                                video_input_dir = gr.Video(lable='上传视频\\upload video',source='upload',interactive=True)
                                video_input_dir.style(width=300)
                                with gr.Row(variant='panel'):
                                    aim_fps_checkbox = gr.Checkbox(label="启用输出帧率控制")
                                    aim_fps = gr.Slider(
                                        minimum=1,
                                        maximum=60,
                                        step=1,
                                        label='输出帧率',
                                        value=30,interactive=True)
                                with gr.Row(variant='panel'):
                                    time_range_checkbox = gr.Checkbox(label="启用时间段裁剪")
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
                                ## 帧生成'视频'/frame2video
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
                    with gr.TabItem(label='ISNETpro'):
                        with gr.Row(variant='panel'):
                            with gr.Column(variant='panel'):
                                gr.Markdown(""" 
                                ## 图片背景去除
                                """)
                                IS_frame_input_dir = gr.Textbox(label='图片输入地址\\frame_input_dir',lines=1,placeholder='input\\folder')
                                IS_frame_output_dir = gr.Textbox(label='图片输出地址\\frame_output_dir',lines=1,placeholder='output\\folder')
                                # with gr.Tabs():
                                #     with gr.TabItem(label='透明背景\\alpha_channel'):
                                #         IS_mode = '透明背景\\alpha_channel'
                                #         IS_bcgrd_dir = "0" 
                                #         IS_rgb_input = "255,255,255"

                                #     with gr.TabItem(label="白色背景\\white_background"):
                                #         IS_mode = "白色背景\\white_background"
                                #         IS_bcgrd_dir = "0"    
                                #         IS_rgb_input = "255,255,255"

                                #     with gr.TabItem(label="纯色背景\\Solid_Color_Background"):
                                #         IS_mode = "纯色背景\\Solid_Color_Background"
                                #         IS_bcgrd_dir = "0"
                                #         IS_rgb_input = gr.Textbox(lable='目标RGB',value='100,100,100',visible=True)

                                #     with gr.TabItem(label="自定义背景\\self_design_Background"):
                                #         IS_mode = "自定义背景\\self_design_Background"
                                #         IS_bcgrd_dir = gr.Textbox(lable='背景图片地址',lines=1,placeholder='input/folder',visible=True)
                                #         IS_rgb_input = "255,255,255"


                                #     with gr.TabItem(label="固定背景\\fixed_background"):
                                #         IS_mode = "固定背景\\fixed_background"
                                #         IS_bcgrd_dir = gr.Textbox(lable='背景图片地址',lines=1,placeholder='input/folder',visible=True)
                                #         IS_rgb_input = "255,255,255"

                                IS_mode = gr.Dropdown(
                                    label="图片输出模式\\frame output mode",
                                    choices=[
                                        "透明背景\\alpha_channel",
                                        "白色背景\\white_background",
                                        "纯色背景\\Solid_Color_Background",
                                        "自定义背景\\self_design_Background",
                                        "固定背景\\fixed_background"],
                                    value="白色背景\\white_background")
                                # if IS_mode ==  "纯色背景\\Solid_Color_Background":
                                # elif IS_mode == "自定义背景\\self_design_Background" or IS_mode == "固定背景\\fixed_background":
                                # else:
                                #     IS_rgb_input = gr.Textbox(lable='目标RGB地址',value='100,100,100',visible=False)
                                #     IS_bcgrd_dir = gr.Textbox(lable='背景图片地址',lines=1,placeholder='input/folder',visible=False)
                                IS_recstrth = gr.Slider(
                                    minimum=7,
                                    maximum=13,
                                    step=1,
                                    label="识别强度\\Recognition strength",
                                    value=10)
                                IS_btn = gr.Button(value="开始生成\\gene_frame")

                            with gr.Column(variant='panel'):
                                gr.Markdown(""" 
                                ## 可选信息填写
                                下面两个请根据自己情况填写，纯色背景的时候需要填写目标RGB，自定义背景和固定背景需要填写背景图片地址  
                                固定背景默认文件夹中的第一张图片  
                                """)
                                IS_rgb_input = gr.Textbox(label="目标RGB",value='100,100,100',visible=True)
                                IS_bcgrd_dir = gr.Textbox(label="背景图片地址\\background_input_dir",lines=1,placeholder='input\\folder',visible=True)
                                with gr.Column(variant='panel'):
                                    IS_out1 = gr.Textbox(label="log info",interactive=False,visible=True,placeholder="output log")

                                IS_btn.click(pic_generation,inputs=[IS_mode,IS_frame_input_dir,IS_bcgrd_dir,IS_frame_output_dir,IS_rgb_input,IS_recstrth],outputs=IS_out1)
                    with gr.TabItem(label='InventColor'):
                        with gr.Row(variant='panel'):
                            with gr.Column(variant='panel'):
                                gr.Markdown(""" 
                                ## 图片反色
                                """)
                                IS_frame_input_dir = gr.Textbox(label='图片输入地址\\frame_input_dir',lines=1,placeholder='input\\folder')
                                IS_frame_output_dir = gr.Textbox(label='图片输出地址\\frame_output_dir',lines=1,placeholder='output\\folder')
                            inv_btn = gr.Button(value="开始生成\\gene_frame")
                            inv_btn.click(fn=ui_invert_image,inputs=[IS_frame_input_dir,IS_frame_output_dir])
                            
                                

                                    




    return [(pro_interface, "isnet_Pro", "isnet_Pro")]