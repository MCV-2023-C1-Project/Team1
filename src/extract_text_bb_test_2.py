import easyocr
import cv2
import numpy as np
import os
import pickle
from sklearn.metrics import jaccard_score
from scipy.signal import convolve2d

reader = easyocr.Reader(['en']) # this needs to run only once to load the model into memory

def scharr_filter(grayscale_img, sobel):
  # set the kernel size, depending on whether we are using the Sobel
  # operator of the Scharr operator, then compute the gradients along
  # the x and y axis, respectively
  #ksize = -1 if args["scharr"] > 0 else 3
  if sobel:
    ksize=3
  else:
    ksize=-1
  gX = cv2.Sobel(grayscale_img, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=ksize, delta=1)
  gY = cv2.Sobel(grayscale_img, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=ksize, delta=1)
  # the gradient magnitude images are now of the floating point data
  # type, so we need to take care to convert them back a to unsigned
  # 8-bit integer representation so other OpenCV functions can operate
  # on them and visualize them
  gX = cv2.convertScaleAbs(gX)
  gY = cv2.convertScaleAbs(gY)
  # combine the gradient representations into a single image
  combined = cv2.addWeighted(gX, 0.5, gY, 0.5, 0)
  combined = cv2.threshold(combined, 125, 255, cv2.THRESH_BINARY)[1]
  return combined


def estimate_noise(I):

    H, W = I.shape

    M = [[1, -2, 1],
       [-2, 4, -2],
       [1, -2, 1]]

    sigma = np.sum(np.sum(np.absolute(convolve2d(I, M))))
    sigma = sigma * np.sqrt(0.5 * np.pi) / (6 * (W-2) * (H-2))

    return sigma


def remove_background(image):
    rgb_planes = cv2.split(image)
    result_norm_planes = []

    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((7, 7), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        norm_img = cv2.normalize(diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        result_norm_planes.append(norm_img)

    enhanced = cv2.merge(result_norm_planes)
    return enhanced


def CDenoissing(image):
    
    (r, g, b) = cv2.split(image)

    b_blur = cv2.medianBlur(b, 3)
    g_blur = cv2.medianBlur(g, 3)
    r_blur = cv2.medianBlur(r, 3)

    b_denoise = cv2.fastNlMeansDenoising(b_blur, 5, 9, 21)
    g_denoise = cv2.fastNlMeansDenoising(g_blur, 5, 9, 21)
    r_denoise = cv2.fastNlMeansDenoising(r_blur, 5, 9, 21)

    enhanced = cv2.merge((r_denoise, g_denoise, b_denoise))
    
    
    return enhanced

IDs = list()

im_list = list()
for im in os.listdir('data/qst2_w3'):
  if '.jpg' in im:
    im_list.append(im)

im_list.sort()

for im in im_list:
  
    if '.jpg' in im:
      imatge = cv2.imread(f'data/qst2_w3/{im}')
      gray = cv2.cvtColor(imatge, cv2.COLOR_BGR2GRAY)

      im_num = im.split('.')[0]

      f = open(f'Results_txt_qst2/{im_num}.txt', 'w')

      mask = cv2.imread(f'data/masks/{im_num}.png')
      mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
      
      if estimate_noise(gray) > 5:
          imatge = CDenoissing(imatge)
          gray = cv2.cvtColor(imatge, cv2.COLOR_BGR2GRAY)


      contours_2 = cv2.findContours(mask_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
      coords = list()
      quadres=list()
      for c in contours_2:
          #Get the BB rectangle
          x, y, w, h = cv2.boundingRect(c)
          quadres.append(imatge[y:y+h, x:x+w, :])
          coords.append((x,y,w,h))

      mask = np.ones(imatge.shape[:2], dtype="uint8") * 255
      for quadre, coord in zip(quadres,coords):
        quadre_gray = cv2.cvtColor(quadre, cv2.COLOR_BGR2GRAY)
        comb = scharr_filter(quadre_gray, True)
        
        # Remove noise
        kernel = np.ones((3,3))
        binary = cv2.morphologyEx(comb, cv2.MORPH_OPEN, kernel)

        #Closing with horizontal kernel to connect the letters
        closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((1,int(quadre.shape[1]/20))))

        # Remove noise
        kernel = np.ones((3,3))
        binary = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)

        #Closing with bigger kernel to connect the letters
        closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((1,int(quadre.shape[1]/6))))

        # find contours
        contours = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

        bb_count = 0
        mask_quadre = np.ones(quadre.shape[:2], dtype="uint8") * 255
        coords_bb = list()

        for c in contours:
            #Get the BB rectangle
            x, y, w, h = cv2.boundingRect(c)
            area = w*h
            if area > 0.002 * quadre.size and (w/h)>3 and (w/h)<20:
              #cv2.rectangle(mask, (coord[0]+x, coord[1]+y), (coord[0]+x+w, coord[1]+y+h), (0, 0, 255), -1)
              cv2.rectangle(mask_quadre, (x, y), (x+w, y+h), (0, 0, 255), -1)
              coords_bb.append((x,y,w,h))
              bb_count+=1
        
        res_final = cv2.bitwise_and(quadre, quadre, mask=cv2.bitwise_not(mask_quadre))

        
        if bb_count>1:
          gray_final = cv2.cvtColor(res_final, cv2.COLOR_BGR2GRAY)
          gX = cv2.Sobel(gray_final, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=3)
          gX = cv2.convertScaleAbs(gX)
          binary_gX = cv2.threshold(gX, 222, 255, cv2.THRESH_BINARY)[1]
          
          # Remove noise
          kernel = np.ones((3,3))
          binary_gX = cv2.morphologyEx(binary_gX, cv2.MORPH_OPEN, kernel)

          #Closing to connect the letters
          closing = cv2.morphologyEx(binary_gX, cv2.MORPH_CLOSE, np.ones((1,int(quadre.shape[1]/10))))

          contours_2 = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
          mask_2 = np.ones(imatge.shape[:2], dtype="uint8") * 255
          mask_quadre_2 = np.ones(quadre.shape[:2], dtype="uint8") * 255
          for c in contours_2:
              #Get the BB rectangle
              convexHull = cv2.convexHull(c)
              x, y, w, h = cv2.boundingRect(convexHull)
              area = w*h
              if area > 0.002 * quadre.size and (w/h)>3 and (w/h)<17:
                cv2.rectangle(mask, (coord[0]+x, coord[1]+y), (coord[0]+x+w, coord[1]+y+h), (0, 0, 255), -1)
                cv2.rectangle(mask_quadre_2, (x, y), (x+w, y+h), (0, 0, 255), -1)

          res_quadre = cv2.bitwise_and(quadre, quadre, mask=cv2.bitwise_not(mask_quadre_2))
          result = reader.readtext(res_quadre, detail=0)
          result_conc = ' '.join(result)
          f.write(result_conc)
          f.write('\n')

        else:
          if coords_bb!=[]:
            mask_quadre_2 = np.ones(quadre.shape[:2], dtype="uint8") * 255  
            cv2.rectangle(mask, (coord[0]+coords_bb[0][0], coord[1]+coords_bb[0][1]), (coord[0]+coords_bb[0][0]+coords_bb[0][2], coord[1]+coords_bb[0][1]+coords_bb[0][3]), (0, 0, 255), -1)
            cv2.rectangle(mask_quadre_2, (coords_bb[0][0], coords_bb[0][1]), (coords_bb[0][0]+coords_bb[0][2], coords_bb[0][1]+coords_bb[0][3]), (0, 0, 255), -1)
            res_quadre = cv2.bitwise_and(quadre, quadre, mask=cv2.bitwise_not(mask_quadre_2))
            result = reader.readtext(res_quadre, detail=0)
            result_conc = ' '.join(result)
            f.write(result_conc)
            f.write('\n')
        

      res_final = cv2.bitwise_and(imatge, imatge, mask=cv2.bitwise_not(mask))

      contours_mask = cv2.findContours(cv2.bitwise_not(mask), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
      aux_list=list()
      for c in contours_mask:
          #Get the BB rectangle
          x, y, w, h = cv2.boundingRect(c)
          brx = x+w
          bry = y+h
          aux_list.append((x,y,brx,bry))
      IDs.append(aux_list)

      cv2.imwrite(f'Pruebas_results_test_2/{im}',res_final)
      cv2.imwrite(f'Masks_text_qst2_w3/{im}', mask)

with open('text_boxes.pkl', 'wb') as file: 
      
  # A new file will be created 
  pickle.dump(IDs, file)
