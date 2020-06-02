import cv2,numpy as np
from skimage import io
import gui
from matplotlib import pyplot as plt

CLIENT_ID='tq3Jki2DtzIL1_kXhvq3'
CLIENT_SECRET='eIasCydQys'

global original
global check
check=False
def search_book(query,display,start):

    from urllib.request import Request, urlopen

    from urllib.parse import urlencode, quote

    import json

    
    request = Request('https://openapi.naver.com/v1/search/book?start='+start+'&'+'display='+display+'&'+'query='+quote(query))

    request.add_header('X-Naver-Client-Id', CLIENT_ID)

    request.add_header('X-Naver-Client-Secret', CLIENT_SECRET)


    response = urlopen(request).read().decode('utf-8')

    search_result = json.loads(response)

    return search_result

def compare_img(img1,img2):
    global check
    ratio=0.7
    MIN_MATCH=10
    FLANN_INDEX_LSH=6
    index_params=dict(algorithm=FLANN_INDEX_LSH,
                  table_number=6,
                  key_size=12,
                  multi_prove_level=1)

    search_params=dict(cheks=32) 
    matcher=cv2.FlannBasedMatcher(index_params,search_params)
    
    results={}
    orb = cv2.ORB_create()
 
    kp1, des1 = orb.detectAndCompute(img1,None)
    kp2, des2 = orb.detectAndCompute(img2,None)
    try:
        matches = matcher.knnMatch(des1,des2,2)
    except:
        return
    good_matches=[m[0] for m in matches\
                 if len (m)==2 and m[0].distance<m[1].distance*ratio]
        
 
    if len(good_matches)>MIN_MATCH:
        src_pts=np.float32([kp1[m.queryIdx].pt for m in good_matches])
        dst_pts=np.float32([kp2[m.trainIdx].pt for m in good_matches])
        mtrx,mask=cv2.findHomography(src_pts,dst_pts,cv2.RANSAC,5.0)
        accuracy=float(mask.sum())/mask.size
    else:
        accuracy=0
    if accuracy*100>60:   
        check=True
        print("책을 찾았습니다!")
        img3 = cv2.drawMatches(img1,kp1,img2,kp2,good_matches[:],None,flags=2)
        
        mngr=plt.get_current_fig_manager()
        mngr.window.setGeometry(300,300,600,500)
        plt.imshow(img3),plt.show()
        
    return accuracy*100
   
def search(img1,books):
    global original
    for book in books:
        terminate=book['image'].index('?type')   
        original=io.imread(book['image'][:terminate])
        original=cv2.resize(original,dsize=(420,440))
        img2=io.imread(book['image'][:terminate])
        img2=cv2.resize(img2,dsize=(img1.shape[1],img1.shape[0]))

        img2=cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

        
        print(book['title'])
        print(compare_img(img1, img2))
        if check==True:
            gui.openImgWin(original, book)
            break    

def printInfo(book):
    print('책 제목: {}'.format(book['title']))
    print('작가명: {}'.format(book['author']))
    print('출간일: {}'.format(book['pubdate']))
    print('가격: {}'.format(book['price']))
    publisher=book['publisher'].replace('<b>','').replace('</b>','')
    print('출판사: {}'.format(publisher))
    desc=book['description'].replace('&#x0D;','')
    print('개요: {}'.format(desc))
    print('-------------------------')

def camcam():
    cap=cv2.VideoCapture(0)
    qImg=None
    while cap.isOpened():
        ret,frame=cap.read()
        if not ret:
            print('No NO No!!')
            break
        h,w=frame.shape[:2]
        left=w//3
        right=(w//3)*2
        top=(h//2)-(h//3)
        bottom=(h//2)+(h//3)
        cv2.rectangle(frame,(left,top),(right,bottom),(255,255,255),3)
    
        flip=cv2.flip(frame,1)
        cv2.imshow('Boo',flip)
        key=cv2.waitKey(10)
        if key==ord(' '):
            qImg=frame[top:bottom,left:right]
            cv2.destroyAllWindows()
            return qImg
            break
        elif key==27:
            break
    else:
        print('No Cam')
    cap.release()
    
    
img1=camcam()
img1=cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)


if __name__ == "__main__":
    books1=[]
    books2=[]
    books3=[]
    startI=1
    
    for i in range(5):
        books1.append(search_book('민음사','100',str(startI))['items'])
        books2.append(search_book('문학동네','100',str(startI))['items'])
        books3.append(search_book('열린책들','100',str(startI))['items'])
        startI+=100
    
        
    circle=0
    while circle<5:
        if check==False:
            search(img1,books1[circle])
        circle+=1
        
    circle=0
    while circle<5:
        if check==False:
            search(img1,books2[circle])
        circle+=1
            
   
       
        
