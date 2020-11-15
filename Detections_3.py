from PIL import Image
import cv2
import numpy as np
from scipy import ndimage
import os
import time

def img_divide(img):
    
    HSV=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    color = [
            ([35, 43,46], [155, 255, 255])  # 蓝色范围，根据测试图的总体颜色，选择用蓝色作为背景与工件的区分
            ]
    for (lower, upper) in color:
      
        lower_co = np.array(lower, dtype="uint8")  
        upper_co = np.array(upper, dtype="uint8") 
        # 根据阈值找到对应颜色
        mask = cv2.inRange(HSV, lower_co, upper_co)    #查找处于范围区间的
        mask = 255-mask #留下工件区域
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours
 
def img_cut(img,contours):
    maxArea = 0
    #寻找工件范围的各坐标点
    for numcontours, contour in enumerate(contours):
        if (cv2.contourArea(contour )>maxArea):
                maxArea = cv2.contourArea( contour )
                x, y, w, h = cv2.boundingRect(contour)

    img_cut = img[int(y):int((y+h)),int(x): int((x+ w))]
    x_width=w
    y_height=h
    return img_cut,x_width,y_height
    

def contour_area(img,contours):
    area = map(cv2.contourArea, contours)
    area_list = list(area)
    area_max = max(area_list)
    post = area_list.index(area_max)#只显示零件外轮廓
    img_cont = np.zeros_like(img)
    img_cont[:, :, :] = 255
    cv2.drawContours(img_cont, contours, post, color=(0,0,0), thickness=-1) # 将零件区域像素值设为(0, 0, 0)
    return img_cont
    
def averange_color(image):
    height, width = image.shape[:2]
    sumc=0
    num=0
    for h in range( height):
        for w in range(width):
            c=image[h,w]
            sumc=sumc+c
            num+=1

    c=int(sumc/num)
    
    return c

def change_color(image,color):
    rows,cols = image.shape[:2]
    for i in range(rows):
      for j in range(cols):
        if image[i,j]==0: # 像素点为255表示的是白色，我们就是要将白色处的像素点，替换为红色
          image[i,j]=color # 
                
    return image
# 对数变化灰度处理
def log_transformation(img):
    A = 0.7
    B = 1.2
    mask = np.ones((3, 3))
    mask[1, 1] = 0

    average_image = ndimage.generic_filter(img.astype(np.float32), np.nanmean, footprint=mask, mode='constant', cval=np.NaN)
    negative_image = 255 - img
    log_image = np.log(negative_image)
    final_log = A*average_image + B*(log_image - average_image)
    return cv2.convertScaleAbs(final_log)


#移除比设定大小小的杂点
def component(image, connectivity=8, min_size=100, show=False):
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity= connectivity)
    sizes = stats[1:, -1]
    nb_components = nb_components - 1
    img = np.zeros((output.shape))
    for i in range(0, nb_components):
        if sizes[i] >= min_size:
            img[output == i + 1] = 255
    img = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    if show:
        cv2.imshow(f'Min size: {min_size}', img)
    return img

# 边缘处理
def approx_cont(contours, point):
    contour = []
    for n in range(len(contours)):
        cont = []
        cont_back = np.concatenate([contours[n][-point:], contours[n]])
        for i in range(len(cont_back[:-point])):
            inner_list = []
            slice = cont_back[i: i + (2*point)].mean(axis=0)
            inner_list.append(int(slice[0][0]))
            inner_list.append(int(slice[0][1]))
            cont.append(inner_list)
        contour.append(np.array(cont).reshape((-1, 1,2)).astype(np.int32))
        
     
    return contour

# 利用morphologyEx（）进行膨胀与腐蚀操作使边缘闭合
def morph(image, kernel, show):
    element_closing = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel)
    img = cv2.morphologyEx(image, cv2.MORPH_CLOSE, element_closing)
    if show:
        cv2.imshow(f'Kernel: {kernel}', img)
    return img

def type_check(cont,width):
    s_max=0 
    c_type=0
    if cont :
        rec_w=[]
        for c in cont:
            rec=cv2.minAreaRect(c)
            w=rec[1][0]
            h=rec[1][1]
            rec_w.append(w)
            s=w*h
            if s>s_max:
                s_max=s
                if w>h:
                    propo=w/h
                else:
                    propo=h/w
        rec_w.sort(reverse=True)
        if propo>8:
            if len(rec_w)>=2:
                if rec_w[0]>=0.8*width:

                    c_type=1
                else:
                    c_type=2
            else:
                c_type=2
        else:
            c_type=3
    
    return c_type
        
        
time_start=time.clock()           
path = 'img/test/'
imagelist = os.listdir(path)
i=1
j=1
z=1
for imgname in imagelist:
 
    if(imgname.endswith(".jpg")):
        
 
        image = cv2.imread(path+imgname)
        #读取图片，并将图片内部的工件分离出来
        #image = cv2.imread("detect15.jpg")
        height, width = image.shape[:2]
        size = (int(width * 0.3), int(height * 0.3))
        image = cv2.resize(image, size)   
        image_cont=img_divide(image)
        image_cut,x_width,y_height=img_cut(image,image_cont)
        #cv2.imshow("Image after transformation to cut", image_cut)
        
        image_cut_cont=img_divide(image_cut.copy())
        image_binary=contour_area(image_cut.copy(),image_cut_cont)
        image_bin_gray=cv2.cvtColor(image_binary, cv2.COLOR_BGR2GRAY)
        #cv2.imshow("Image after transformation to binary", image_bin_gray)
        image_cut_area=cv2.bitwise_or(image_cut,image_binary)
       # cv2.imshow("Image 1", image_cut_area)
        
        #灰度转换
        image_gray = cv2.cvtColor(image_cut_area, cv2.COLOR_BGR2GRAY)
       # cv2.imshow("Image after transformation to gray", image_gray)


        color_change=averange_color(image_gray)
        image_binary_reverse=255-image_bin_gray
        image_change=change_color(image_binary_reverse,color_change)
        #cv2.imshow("Image 2", image_change)
        image_ave_cont=cv2.bitwise_and(image_change,image_gray)
        #cv2.imshow("Image 3", image_ave_cont)
        
  
       
        
        # 对数变换
        image_log = log_transformation(image_ave_cont)
        #cv2.imshow("Image after Logarithm trans", image_log)

        #双边滤波去噪
 #       image_bilateral = cv2.bilateralFilter(image_log, 1, 20, 20)
  #      cv2.imshow("Image after Bilateral Filter", image_bilateral)

        #高斯滤波去噪
        image_gaussianblur=cv2.GaussianBlur(image_log, (5, 5), 0)
        #cv2.imshow("Image after GaussianBlur", image_gaussianblur)

        # 归一化函数
        image_normalize = cv2.normalize(image_gaussianblur, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        #cv2.imshow("Image after normalize", image_normalize)
        
        
        
        #边缘检测
        image_canny = cv2.Canny(image_normalize, 80, 100)
        #cv2.imshow("Image after Canny transformation", image_canny)
        
        bin_cont, bin_hierarchy = cv2.findContours(image_bin_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        image_bin_cont = cv2.drawContours(image_bin_gray, bin_cont ,-1, (255,255,255) ,7)
        #cv2.imshow("Image after bincont", image_bin_cont)
        image_bin_mask=255-image_bin_cont
        image_cont_remove=cv2.bitwise_and(image_canny,image_canny,mask=image_bin_mask)
        #cv2.imshow("Image after removing contours", image_cont_remove)
        
      
        #腐蚀膨胀操作
        image_morph = morph(image_cont_remove , kernel=(5,5), show=False)
        #cv2.imshow("Image after morphologyEx", image_morph)

        #移除杂点
        image_conn = component(image_morph, min_size= 120, show=False)
        #cv2.imshow("Image after removing edges", image_conn)

        #归一化函数
        image_norm = cv2.normalize(image_conn, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

        #轮廓检测
        cont, hierarchy = cv2.findContours(image_conn.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        #处理后的轮廓
        new_cont = approx_cont(cont, point=8)


        #画出处理过的轮廓
        image_cont_approx = cv2.drawContours(image_cut, new_cont ,-1, (0,0,255) ,2)
        #cv2.imshow('Image with approx contours', image_cont_approx)

        c_type=type_check(new_cont,x_width)

        if c_type==1:
            cv2.putText(image,"Non-Hiding", (50,50),cv2.FONT_HERSHEY_SIMPLEX, 2,(0,0, 255), 2) 
        elif c_type==2:
            cv2.putText(image,"Scratch", (50,50),cv2.FONT_HERSHEY_SIMPLEX, 2,(0,0, 255), 2) 
        else:
            cv2.putText(image,"Blistering", (50,50),cv2.FONT_HERSHEY_SIMPLEX, 2,(0,0, 255), 2) 

        # Draw mask on image
        #image_cont_approx_mask = cv2.drawContours(image.copy(), new_cont ,-1, (0,0,255), -1)
        #image_final = cv2.addWeighted(image_cont_approx_mask, 0.5, image, 1 - 0.5, 0, image)
        # cv2.imshow("Image with mask", image_final)

        #还原至原图
        image = Image.fromarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
        image_cont_approx = Image.fromarray(cv2.cvtColor(image_cont_approx,cv2.COLOR_BGR2RGB))
        image.paste(image_cont_approx, (int(x_width),int(y_height)), mask=None)
        image.show()
        if c_type==1:
            image.save('output/Non-Hiding'+str(i)+' '+imgname+'.bmp')
            i+=1
        elif c_type==2:
            image.save('output/Scratch'+str(j)+' '+imgname+'.bmp')
            j+=1
        else:
            image.save('output/Blistering'+str(z)+' '+imgname+'.bmp')
            z+=1
        print("已处理"+str(i+j+z-3)+"张图片")
print("运行结束")
time_end=time.clock()
print("totally cost",time_end-time_start)
cv2.waitKey(0)
