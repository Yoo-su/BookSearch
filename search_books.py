# -*- coding: utf-8 -*-
#네이버 개발자센터로부터 받은 ID, PW
CLIENT_ID='tq3Jki2DtzIL1_kXhvq3'
CLIENT_SECRET='eIasCydQys'

#파라미터로 출판사명, 가져올 데이터 수, 시작 위치를 받음
def search_book(publ,display,start):

    from urllib.request import Request, urlopen

    from urllib.parse import urlencode, quote

    import json

    #요청변수 값을 넣어 검색 요청 (json)
    request = Request('https://openapi.naver.com/v1/search/book_adv?&start='
                      +start+'&'+'display='+display+'&'+'d_publ='+quote(publ))
 
    #허가 위해 ID, PW값 헤더에 추가
    request.add_header('X-Naver-Client-Id', CLIENT_ID)

    request.add_header('X-Naver-Client-Secret', CLIENT_SECRET)

    response = urlopen(request).read().decode('utf-8')

    #json 형식 string을 딕셔너리 형식으로 바꾸어 결과 저장
    search_result = json.loads(response)

    return search_result

