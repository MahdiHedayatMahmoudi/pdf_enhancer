#! /usr/bin/env python3

__version__= '1.0'

import argparse
import sys
import os
import numpy as np
import warnings
import cv2
from keras.models import load_model
import tensorflow as tf
from pdf2image import convert_from_path
from PyPDF2 import PdfFileWriter, PdfFileReader
from fpdf import FPDF
import img2pdf 
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    
__doc__=\
"""
Tool to load model and binarize a given image.
"""

class pdf_enhancer:
    def __init__(self,dir_inp,model,dir_out, patches='false' ):
        self.dir_pdf=dir_inp
        self.patches=patches
        self.dir_out=dir_out
        self.model_dir=model

    def resize_image(self,img_in,input_height,input_width):
        return cv2.resize( img_in, ( input_width,input_height) ,interpolation=cv2.INTER_NEAREST)
    
    def start_new_session_and_model(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True
    
        self.session =tf.Session(config=config)# tf.InteractiveSession()
    def load_model(self):
        self.model = load_model(self.model_dir , compile=False)
        
        
        self.img_height=self.model.layers[len(self.model.layers)-1].output_shape[1]
        self.img_width=self.model.layers[len(self.model.layers)-1].output_shape[2]
        self.n_classes=self.model.layers[len(self.model.layers)-1].output_shape[3]

    def end_session(self):
        self.session.close()


        del self.model
        del self.session
    def predict(self,img_org,bin_scale):
        ##img_org=cv2.imread(self.image)
        img=self.resize_image(img_org,int(img_org.shape[0]/bin_scale),int(img_org.shape[1]/bin_scale))
        img_width_model=self.img_width
        img_height_model=self.img_height

        if self.patches=='true' or self.patches=='True':

            margin = int(0.1 * img_width_model)

            width_mid = img_width_model - 2 * margin
            height_mid = img_height_model - 2 * margin


            img = img / float(255.0)

            img_h = img.shape[0]
            img_w = img.shape[1]

            prediction_true = np.zeros((img_h, img_w, 3))
            mask_true = np.zeros((img_h, img_w))
            nxf = img_w / float(width_mid)
            nyf = img_h / float(height_mid)

            if nxf > int(nxf):
                nxf = int(nxf) + 1
            else:
                nxf = int(nxf)

            if nyf > int(nyf):
                nyf = int(nyf) + 1
            else:
                nyf = int(nyf)

            for i in range(nxf):
                for j in range(nyf):

                    if i == 0:
                        index_x_d = i * width_mid
                        index_x_u = index_x_d + img_width_model
                    elif i > 0:
                        index_x_d = i * width_mid
                        index_x_u = index_x_d + img_width_model

                    if j == 0:
                        index_y_d = j * height_mid
                        index_y_u = index_y_d + img_height_model
                    elif j > 0:
                        index_y_d = j * height_mid
                        index_y_u = index_y_d + img_height_model

                    if index_x_u > img_w:
                        index_x_u = img_w
                        index_x_d = img_w - img_width_model
                    if index_y_u > img_h:
                        index_y_u = img_h
                        index_y_d = img_h - img_height_model
                        
                    

                    img_patch = img[index_y_d:index_y_u, index_x_d:index_x_u, :]

                    label_p_pred = self.model.predict(
                        img_patch.reshape(1, img_patch.shape[0], img_patch.shape[1], img_patch.shape[2]))

                    seg = np.argmax(label_p_pred, axis=3)[0]

                    seg_color = np.repeat(seg[:, :, np.newaxis], 3, axis=2)

                    if i==0 and j==0:
                        seg_color = seg_color[0:seg_color.shape[0] - margin, 0:seg_color.shape[1] - margin, :]
                        seg = seg[0:seg.shape[0] - margin, 0:seg.shape[1] - margin]

                        mask_true[index_y_d + 0:index_y_u - margin, index_x_d + 0:index_x_u - margin] = seg
                        prediction_true[index_y_d + 0:index_y_u - margin, index_x_d + 0:index_x_u - margin,
                        :] = seg_color
                        
                    elif i==nxf-1 and j==nyf-1:
                        seg_color = seg_color[margin:seg_color.shape[0] - 0, margin:seg_color.shape[1] - 0, :]
                        seg = seg[margin:seg.shape[0] - 0, margin:seg.shape[1] - 0]

                        mask_true[index_y_d + margin:index_y_u - 0, index_x_d + margin:index_x_u - 0] = seg
                        prediction_true[index_y_d + margin:index_y_u - 0, index_x_d + margin:index_x_u - 0,
                        :] = seg_color
                        
                    elif i==0 and j==nyf-1:
                        seg_color = seg_color[margin:seg_color.shape[0] - 0, 0:seg_color.shape[1] - margin, :]
                        seg = seg[margin:seg.shape[0] - 0, 0:seg.shape[1] - margin]

                        mask_true[index_y_d + margin:index_y_u - 0, index_x_d + 0:index_x_u - margin] = seg
                        prediction_true[index_y_d + margin:index_y_u - 0, index_x_d + 0:index_x_u - margin,
                        :] = seg_color
                        
                    elif i==nxf-1 and j==0:
                        seg_color = seg_color[0:seg_color.shape[0] - margin, margin:seg_color.shape[1] - 0, :]
                        seg = seg[0:seg.shape[0] - margin, margin:seg.shape[1] - 0]

                        mask_true[index_y_d + 0:index_y_u - margin, index_x_d + margin:index_x_u - 0] = seg
                        prediction_true[index_y_d + 0:index_y_u - margin, index_x_d + margin:index_x_u - 0,
                        :] = seg_color
                        
                    elif i==0 and j!=0 and j!=nyf-1:
                        seg_color = seg_color[margin:seg_color.shape[0] - margin, 0:seg_color.shape[1] - margin, :]
                        seg = seg[margin:seg.shape[0] - margin, 0:seg.shape[1] - margin]

                        mask_true[index_y_d + margin:index_y_u - margin, index_x_d + 0:index_x_u - margin] = seg
                        prediction_true[index_y_d + margin:index_y_u - margin, index_x_d + 0:index_x_u - margin,
                        :] = seg_color
                        
                    elif i==nxf-1 and j!=0 and j!=nyf-1:
                        seg_color = seg_color[margin:seg_color.shape[0] - margin, margin:seg_color.shape[1] - 0, :]
                        seg = seg[margin:seg.shape[0] - margin, margin:seg.shape[1] - 0]

                        mask_true[index_y_d + margin:index_y_u - margin, index_x_d + margin:index_x_u - 0] = seg
                        prediction_true[index_y_d + margin:index_y_u - margin, index_x_d + margin:index_x_u - 0,
                        :] = seg_color
                        
                    elif i!=0 and i!=nxf-1 and j==0:
                        seg_color = seg_color[0:seg_color.shape[0] - margin, margin:seg_color.shape[1] - margin, :]
                        seg = seg[0:seg.shape[0] - margin, margin:seg.shape[1] - margin]

                        mask_true[index_y_d + 0:index_y_u - margin, index_x_d + margin:index_x_u - margin] = seg
                        prediction_true[index_y_d + 0:index_y_u - margin, index_x_d + margin:index_x_u - margin,
                        :] = seg_color
                        
                    elif i!=0 and i!=nxf-1 and j==nyf-1:
                        seg_color = seg_color[margin:seg_color.shape[0] - 0, margin:seg_color.shape[1] - margin, :]
                        seg = seg[margin:seg.shape[0] - 0, margin:seg.shape[1] - margin]

                        mask_true[index_y_d + margin:index_y_u - 0, index_x_d + margin:index_x_u - margin] = seg
                        prediction_true[index_y_d + margin:index_y_u - 0, index_x_d + margin:index_x_u - margin,
                        :] = seg_color

                    else:
                        seg_color = seg_color[margin:seg_color.shape[0] - margin, margin:seg_color.shape[1] - margin, :]
                        seg = seg[margin:seg.shape[0] - margin, margin:seg.shape[1] - margin]

                        mask_true[index_y_d + margin:index_y_u - margin, index_x_d + margin:index_x_u - margin] = seg
                        prediction_true[index_y_d + margin:index_y_u - margin, index_x_d + margin:index_x_u - margin,
                        :] = seg_color

            prediction_true = prediction_true.astype(np.uint8)
            prediction_true=self.resize_image(prediction_true,img_org.shape[0],img_org.shape[1])
                
        else:
            img_h_page=img.shape[0]
            img_w_page=img.shape[1]
            img = img /float( 255.0)
            img = self.resize_image(img, img_height_model, img_width_model)

            label_p_pred = self.model.predict(
                img.reshape(1, img.shape[0], img.shape[1], img.shape[2]))

            seg = np.argmax(label_p_pred, axis=3)[0]
            seg_color =np.repeat(seg[:, :, np.newaxis], 3, axis=2)
            prediction_true = self.resize_image(seg_color, img_h_page, img_w_page)
            prediction_true = prediction_true.astype(np.uint8)
            
            
        return prediction_true[:,:,0]

    def run(self):
        self.start_new_session_and_model()
        self.load_model()
        
        pdf_reader= PdfFileReader(self.dir_pdf)
        num_pages=pdf_reader.getNumPages()
        
        dir_to_write_single_page=self.dir_out+'/single_page.pdf'
        dir_to_write_single_page_image=self.dir_out+'/single_page_image.jpg'
        
        dir_imgs_enhanced=os.path.join(self.dir_out,'images')
        
        if os.path.isdir(dir_imgs_enhanced):
            os.system('rm -rf '+dir_imgs_enhanced)
            os.makedirs(dir_imgs_enhanced)
        else:
            os.makedirs(dir_imgs_enhanced)
        #os.makedirs(dir_imgs_enhanced)

        indexer=0
        #pdf_reader = PdfFileReader('/home/vahid/Documents/en
        for num_page in range(num_pages):
            pdf_writer = PdfFileWriter()
            
            page_single = pdf_reader.getPage(num_page)
            
            pdf_writer.addPage(page_single)
            
            
            with open(dir_to_write_single_page, 'wb') as out:
                pdf_writer.write(out)
            #pdf_writer.write(out)
            pages = convert_from_path(dir_to_write_single_page,'500')
            
            for page in pages:
                page.save(dir_to_write_single_page_image, 'JPEG')
                
                
            img=cv2.imread(dir_to_write_single_page_image)
            
            bin_scales=[1]
            img_last=0
            
            for bin_s in bin_scales:
                res=self.predict(img,bin_s)

                img_fin=np.zeros((res.shape[0],res.shape[1],3) )
                res[:,:][res[:,:]==0]=2
                res=res-1
                res=res*255
                img_fin[:,:,0]=res
                img_fin[:,:,1]=res
                img_fin[:,:,2]=res
                
                img_fin=img_fin.astype(np.uint8)
                img_fin=(res[:,:]==0)*255
                img_last=img_last+img_fin

            img_last[:,:][img_last[:,:]>0]=255
            img_last=(img_last[:,:]==0)*255
            
            cv2.imwrite(os.path.join(dir_imgs_enhanced,'{0:04}'.format(indexer)+'.jpg'),img_last)
            indexer=indexer+1
            
        with open(self.dir_out+"/output.pdf", "wb") as f:
            f.write(img2pdf.convert([dir_imgs_enhanced+'/'+i for i in os.listdir(dir_imgs_enhanced) if i.endswith(".jpg")]))
            
        os.system('rm -rf '+dir_imgs_enhanced)
        
        

def main():
    parser=argparse.ArgumentParser()
    
    parser.add_argument('-in','--input', dest='inp1', default=None, help='pdf file.')
    parser.add_argument('-p','--patches', dest='inp3', default=False, help='by setting this parameter to true you let the model to see the image in patches.')
    parser.add_argument('-out','--output', dest='inp4', default=False, help='directory to write output')
    parser.add_argument('-m','--model', dest='inp2', default=None, help='models directory.')
    
    options=parser.parse_args()
    
    possibles=globals()
    possibles.update(locals())
    x=pdf_enhancer(options.inp1,options.inp2,options.inp4,options.inp3)
    x.run()

if __name__=="__main__":
    main()

    
    
    
