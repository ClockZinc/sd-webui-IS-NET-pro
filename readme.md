# sd-webui-IS-net
在大佬的基础上做了一个简单的插件  
## 大佬论文开源库：
[github|DIS](https://github.com/xuebinqin/DIS)
# 主要功能|functions
## 对图像的处理，以及抠图|image process
### remove background (.png recommended)
#### white background mode
![image](https://user-images.githubusercontent.com/118424801/230847648-901bc3b4-c44c-4d9c-a609-d226019ea5b9.png)
#### alpha
![image](https://user-images.githubusercontent.com/118424801/231087354-6b4fa51b-3be7-419b-a289-f185f0e627fb.png)
#### pure color
![image](https://user-images.githubusercontent.com/118424801/231087560-662881ab-42dc-4c9e-b9a9-d6551fc1bc63.png)
![image](https://user-images.githubusercontent.com/118424801/231087606-a291a14f-c968-496e-8c90-8abb6e1bc3ff.png)
#### self design background
You should create a folder with background images
![image](https://user-images.githubusercontent.com/118424801/231087832-d074f301-5947-4996-9fe2-aabd1add9294.png)
![image](https://user-images.githubusercontent.com/118424801/231088083-59f96d4c-232d-421f-8675-3ebff0208cb5.png)
### generate frames from a video(.mp4 only)
![image](https://user-images.githubusercontent.com/118424801/230847917-33a58f82-1d2d-4af8-bea5-a8a5166856ca.png)
# How to use it
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

