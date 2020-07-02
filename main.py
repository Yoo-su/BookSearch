# -*- coding: utf-8 -*-
import cv2,numpy as np
from skimage import io
from cam import camcam
from gui import openImgWin
from search_books import search_book
import webbrowser
from matplotlib import pyplot as plt
import time

global original
global check
global start

check=False

#sift 기술자 사용방법
def compare_img2(img1,img2,kp1,des1):
    global check
    MIN_MATCH_COUNT = 25
    
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
        
    #SIFT 추출기 생성
    sift = cv2.xfeatures2d.SIFT_create()

    #SIFT 알고리즘으로 키포인트 검출과 특징기술자 계산을 한번에 수행
    kp2, des2 = sift.detectAndCompute(img2,None)
    
    #매칭점 계산, 좋은 매칭점 추출
    matches = flann.knnMatch(des1,des2,k=2)
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)
    if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        accuracy=float(mask.sum())/mask.size
    else:
        accuracy=0
    
    if accuracy*100>70:   
        check=True
        print("책을 찾았습니다!")
        img3 = cv2.drawMatches(img1,kp1,img2,kp2,good[:],None,flags=2)
        
        mngr=plt.get_current_fig_manager()
        mngr.window.setGeometry(300,300,600,500)
        plt.imshow(img3),plt.show()
    
    return accuracy*100


#ORB 기술자 사용방법    
def compare_img(img1,img2,kp1,des1):
    global check
    ratio=0.7
    MIN_MATCH=25
    
    #매칭을 위한 파라미터 값 설정과정
    FLANN_INDEX_LSH=6
    index_params=dict(algorithm=FLANN_INDEX_LSH,
                  table_number=6,
                  key_size=12,
                  multi_prove_level=1)

    search_params=dict(cheks=32) 
    matcher=cv2.FlannBasedMatcher(index_params,search_params)
    
    #ORB추출기 생성
    orb = cv2.ORB_create()
    
    """ORB로 찾으려는 이미지 img1과 비교 이미지 img2 각각에서 키 포인트 검출과
    특징 디스크립터 계산을 한번에 수행하는 과정이다"""
    kp2, des2 = orb.detectAndCompute(img2,None)
    
    #검출한 각 영상의 특징 디스크립터를 knn매치를 사용해 매칭되는 부분을 찾는다.
    try:
        matches = matcher.knnMatch(des1,des2,2)
    except:
        return
    #앞의 것의 거리(distance)가 뒤의 것의 거리의 75% 인 것들만 모아서 good_matches에 보관
    good_matches=[m[0] for m in matches\
                 if len (m)==2 and m[0].distance<m[1].distance*ratio]
        
 
    if len(good_matches)>MIN_MATCH:
        #좋은 매칭점을 통해 입력영상, 비교영상 각각에서의 좌표 구하는 과정
        src_pts=np.float32([kp1[m.queryIdx].pt for m in good_matches])
        dst_pts=np.float32([kp2[m.trainIdx].pt for m in good_matches])
        #원근 변환행렬 구하기
        """원근 변환행렬은 비교 이미지 상에서 입력 이미지와 일치하는 부분이 어디 있는지
        0과 1로 표현한 행렬이다 1이 입력 이미지와 매칭되는 부분"""
        mtrx,mask=cv2.findHomography(src_pts,dst_pts,cv2.RANSAC,5.0)
        #원근변환행렬에서 1의 개수(매칭점 개수)를 행렬 크기로 나눠서 유사도를 구한다
        accuracy=float(mask.sum())/mask.size
    else:
        accuracy=0
    if accuracy*100>70:   
        check=True
        print("책을 찾았습니다!")
        img3 = cv2.drawMatches(img1,kp1,img2,kp2,good_matches[:],None,flags=2)
        
        mngr=plt.get_current_fig_manager()
        mngr.window.setGeometry(300,300,600,500)
        plt.imshow(img3),plt.show()
        
    return accuracy*100
   
def search(img1,books):
    global original
    global start
    orb = cv2.ORB_create()
    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = orb.detectAndCompute(img1,None)
    
    for book in books:
        terminate=book['image'].index('?type')   
        original=io.imread(book['image'][:terminate])
        original=cv2.cvtColor(original,cv2.COLOR_BGR2RGB)
        original=cv2.resize(original,dsize=(420,440))
        
        img2=io.imread(book['image'][:terminate])
        img2=cv2.resize(img2,dsize=(img1.shape[1],img1.shape[0]))
        img2=cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)


        winname = "test"
        #cv2.namedWindow(winname)   # create a named window
        #cv2.moveWindow(winname, 720, 250)   # Move it to (40, 30)
        #cv2.imshow(winname, original)
        #cv2.waitKey(50)
        #cv2.destroyAllWindows()
        print(book['title'])
        print(compare_img(img1,img2,kp1,des1))
        if check==True:
            webbrowser.open(book['link'])
            print("검색에 소요된 시간: "+str(time.time()-start)+"초")
            openImgWin(original, book)
  
            break
  


def InputStart(books):
    global start
    start=time.time()
    img1=camcam()
    img1=cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
 
    circle=0
    while circle<len(books):
        if check==False:
            search(img1,books[circle])
        circle+=1
   
    
if __name__ == "__main__":
    global start
    books=[]
    startI=1
    
    for i in range(5):
        books.append(search_book('민음사','100',str(startI))['items'])
        books.append(search_book('문학동네','100',str(startI))['items'])
        books.append(search_book('열린책들','100',str(startI))['items'])
        books.append(search_book('창비','100',str(startI))['items'])
        books.append(search_book('위즈덤하우스','100',str(startI))['items'])
        startI+=100
    
    InputStart(books)
    
    
        