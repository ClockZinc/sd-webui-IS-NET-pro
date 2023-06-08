import cv2
import os, re
import numpy as np
from tqdm import tqdm
def ui_frame2video(image_folder,ouput_dir,fps,mode):
    print("\n IS-NET_pro:frame2video generating...")
    if mode =='.mp4':
        return frame2video(image_folder,ouput_dir,fps)
    elif mode == '.avi':
        return frame2video_alpga(image_folder,ouput_dir,fps)
    
def check_chinese(path):
    pattern = re.compile(u'[\u4e00-\u9fa5]') # 匹配中文字符的正则表达式
    if re.search(pattern, path):
        raise ValueError("输出路径含有中文，输出失败")


def video2frame(video_path,output_folder,aim_fps_checkbox,aim_fps,time_range_checkbox,start_time,end_time):
    """
    考虑一下总的生成逻辑
    帧拆解就是考虑这个帧率和时间
    最终生成frame_indexes即可
    起始与终点时间为帧率乘以起点与终点时间，这取决于是否开启时间范围控制。
    生成的帧率总数为时间范围乘以帧率，帧率总数为int( total_frames * aim_fps / video_fps)
    np.linspace(max(start_frame,0), min(total_frames - 1,end_frame - 1 ), min(int( (end_time-start_time) * aim_fps),end_frame), dtype=np.int)

    """
    
    check_chinese(output_folder)  

    print("\n IS-NET_pro:video2frame generating...")

    # 读取视频文件
    cap = cv2.VideoCapture(video_path)

    # 检查视频是否成功打开
    if not cap.isOpened():
        print("Error opening video file")

    # 创建输出文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 首先确定总的帧率是多少,视频原帧率为
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    # 判断帧率是否需要改变
    if aim_fps_checkbox:
        # 这个问题是关于视频转图片的方法。首先，我们需要知道视频的帧率（fps），
        # 即每秒钟播放的帧数。然后，我们可以计算每个输出图片之间的时间间隔，即 
        # 1/fps。接着，我们需要确定每个输出图片所在的时间点。为了做到这一点，我
        # 们可以将输出图片的序号乘以时间间隔，然后将结果乘以视频的帧率，再向下取整
        # ，就可以得到输出图片所对应的视频帧。最后，我们只需要将这些视频帧保存为图片即可。
        # 这是视频的总帧率
        total_output_frames = int( total_frames * aim_fps / video_fps)
    else:
        total_output_frames = total_frames


    # 计算起始帧与终止帧的index，这边还是使用原帧数才是正确的
    if time_range_checkbox:
        start_frame = int(start_time * video_fps)
        end_frame = int(end_time * video_fps)
    else :
        start_frame = 0
        end_frame = total_frames

    # 生成需要输出的帧的索引
    start_frame = max(start_frame, 0)
    end_frame   = min(end_frame, total_frames)

    # 输出的总帧数 = 持续时长乘以视频的总帧率
    # 如果需要改变帧率，乘以目标帧率
    if aim_fps_checkbox and time_range_checkbox:
        output_frames = int( (end_time-start_time) * aim_fps)
    elif time_range_checkbox and (not aim_fps_checkbox):
        output_frames = int( (end_time-start_time) * video_fps)
    else :
        output_frames = int(total_output_frames)

    frame_indexes = np.linspace(start_frame, end_frame - 1, output_frames, dtype=int)

    if aim_fps_checkbox or time_range_checkbox:
        # 这部分是根据frame_indexes生成视频帧的部分不需要改动吧
        frame_count = 1
        for i in tqdm(frame_indexes):
        # 设置读取帧的位置
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            # 读取帧并保存为图片
            ret, frame = cap.read()
            if ret:
                # 指定输出文件名
                output_file = os.path.join(output_folder, f'{frame_count:04d}.png')
                # print('\r\n geneframe:',output_file,end='')

                # 保存帧到输出文件
                cv2.imwrite(output_file, frame)
                frame_count += 1
    else :
        frame_count = 1
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for i in tqdm(range(num_frames)):
            # 读取一帧
            ret, frame = cap.read()
            # 检查是否成功读取帧
            if not ret:
                break
            else:
            # 指定输出文件名
                output_file = os.path.join(output_folder, f'{frame_count:04d}.png')
                # print('\r geneframe:',output_file,end='')

                # 保存帧到输出文件
                cv2.imwrite(output_file, frame)

                # 更新帧计数器
                frame_count += 1

    

    # 释放视频对象
    cap.release()
    print('\n:) done!')

    return ":) done"
    
def frame2video(image_folder,ouput_dir,fps):
    # 读取图像文件列表
    image_files = [f for f in os.listdir(image_folder) if f.endswith('.png') or f.endswith('.jpg')]
    image_files.sort()

    # 获取图像的宽度和高度
    img = cv2.imread(os.path.join(image_folder, image_files[0]),cv2.IMREAD_UNCHANGED)
    height, width, _ = img.shape

    # 创建输出视频对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(ouput_dir+'/output.mp4', fourcc, fps, (width, height), isColor=True)
    num_images = len(image_files)
    frame_num = 0
    # 逐帧写入视频帧
    for image_file in tqdm(image_files):
        image_path = os.path.join(image_folder, image_file)
        frame = cv2.imread(image_path)
        out.write(frame)
        frame_num +=1
        # print('\r generating video:',f'{100*frame_num/num_images:5.2f}%',end='')

    # 释放视频对象
    out.release()
    print('\n:) done!')
    return ":) done"


def frame2video_alpga(image_folder,ouput_dir,fps):
    # 读取图像文件列表
    image_files = [f for f in os.listdir(image_folder) if f.endswith('.png') or f.endswith('.jpg')]
    image_files.sort()

    # 获取图像的宽度和高度
    img = cv2.imread(os.path.join(image_folder, image_files[0]),cv2.IMREAD_UNCHANGED)
    height, width, _ = img.shape

    # 创建输出视频对象
    # 格式表在这里：自己查一下对照表
    # https://learn.microsoft.com/en-us/windows/win32/medfound/video-fourccs
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(ouput_dir+'/output.avi', fourcc, fps, (width, height), isColor=True)
    num_images = len(image_files)
    frame_num = 0
    # 逐帧写入视频帧
    for image_file in tqdm(image_files):
        image_path = os.path.join(image_folder, image_file)
        frame = cv2.imread(image_path)
        out.write(frame)
        frame_num +=1
        # print('\r generating video:',f'{100*frame_num/num_images:5.2f}%',end='')

    # 释放视频对象
    out.release()
    print('\n:) done!')
    return ":) done"

if __name__ == '__main__':
    # image_folder = r"D:\Doctoral_Career\Little_interest\novelAI\SD_img2img_Video\test\course2\output4"
    # ouput_dir = r"D:\Doctoral_Career\Little_interest\novelAI\SD_img2img_Video\test\course2\output4"
    # fps = 30
    video2frame("D:\Doctoral_Career/Little_interest/AI_animation/animation_project/测试isnet_pro/video_src/星瞳神级现场系列丨⭐两个人⭐点击获得美丽双子星！.mp4",
                r"D:\Doctoral_Career\Little_interest\AI_animation\animation_project\test_isnet_pro\frame",True,15,True ,0,1)