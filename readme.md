# sd-webui-IS-net
在大佬的基础上做了一个简单的插件  
Just  git it in your extensions folder in your Stable-Diffusion-WebUI  
```
git clone https://github.com/ClockZinc/sd-webui-IS-NET-pro
```
## 大佬论文开源库：
[github|DIS](https://github.com/xuebinqin/DIS)
# 主要功能|functions
## 对图像的处理，以及抠图|image processing
### Stingy picture
My plugin can batch generate images as well as generate a single image. In single image mode, the result will be displayed directly. 
![image](https://user-images.githubusercontent.com/118424801/232211245-a8e8d610-79eb-45b4-add6-8bffa990751d.png)
## Masked Multi Frame Render
In img2img mode, at the bottom of the page, open "ISnet::MFR".
![image](https://user-images.githubusercontent.com/118424801/232223011-718bea30-713c-4357-a195-611a9163c745.png)
![image](https://user-images.githubusercontent.com/118424801/232223031-7629a917-7d2a-4bfa-ae50-2da8102ed0e1.png)
It can batch generate image using inpaint
Which can only modify the masked area. It will also generates a mask folder.
![image](https://user-images.githubusercontent.com/118424801/232223133-48a72a1a-0b71-43c9-980d-5aa88f69fd21.png)
# How to use it
New functions in New Version [BV1nk4y1e76X](https://www.bilibili.com/video/BV1nk4y1e76X)  
A bilibili video as follows[BV1Fh411G7dw](https://www.bilibili.com/video/BV1Fh411G7dw)
## example
input folder can be anywhere in you device. It should be like D:\path\to\folder .  
![image](https://user-images.githubusercontent.com/118424801/230843907-9432dc93-ac32-4846-bc85-4a80014bfe99.png)
as shown in follows, the folder has a iamge(can be more), I fill in the path of it  
![image](https://user-images.githubusercontent.com/118424801/230844923-2343a923-b9cf-43c2-8aa2-904faf70a60e.png)
The output folder, has nothing in it(it could have some)  
![image](https://user-images.githubusercontent.com/118424801/230844367-80d4bc33-62d5-4085-a4eb-02ab9d390e23.png)
Then click the gene_frame button  
![image](https://user-images.githubusercontent.com/118424801/230844690-f517e2d5-0ff1-4c09-8e05-fe28d61ad026.png)
Then it generates a image with white background  
![image](https://user-images.githubusercontent.com/118424801/230845020-522d7d80-af1f-4677-9d30-3eace505d390.png)
## model
The extension will create a folder with the structure bellow, and automatically download a `[isnet-general-use.pth](https://huggingface.co/ClockZinc/IS-NET_pth/resolve/main/isnet-general-use.pth)`  
--sd-webui-IS-NET-pro  
  |  
  --saved_models  
    |  
    --IS-Net  
      |  
      --isnet-general-use.pth  
![image](https://user-images.githubusercontent.com/118424801/230846300-34c48248-5c9c-4348-8c8d-f7af90b6c966.png)
![image](https://user-images.githubusercontent.com/118424801/230847128-7f1f6fd8-9aae-4611-9a03-b41f3213903b.png)
If you find out you don't have it in your extension file, you could create it by yourself.

