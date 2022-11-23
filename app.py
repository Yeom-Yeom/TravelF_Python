# This Python file uses the following encoding: utf-8

from flask import Flask,redirect
from flask import request
from urllib.parse import unquote_plus
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)


@app.route("/", methods=['GET'])
def recommend():
    test = pd.read_csv(
        r'D:/TravelF/src/main/webapp/crawling/recommended.csv',
        encoding='utf-8', low_memory=False)
    test = test[['id', 'title', 'file_name', 'hashtag', 'area']]
    test = test.dropna()
    
    test['hashtag'] = test['hashtag'].fillna('')

    test1 = test['area'] == request.args['area'] + ' ' + request.args['sigungu']
    test = test.loc[test1, :]

    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(test['hashtag'])

    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    cosine_sim = pd.DataFrame(cosine_sim, index=test.index, columns=test.index)

    indices = pd.Series(test.index, index=test['title'])

    def content_recommender(title, n_of_recomm):

        # title에서 여행지 index 받아오기
        idx = indices[title]

        # 주어진 여행지와 다른 여행지의 similarity를 가져온다
        sim_scores = cosine_sim[idx]

        # similarity 기준으로 정렬하고 n_of_recomm만큼 가져오기
        sim_scores = sim_scores.sort_values(ascending=False)[1:n_of_recomm + 1]

        # id 반환
        return test.loc[sim_scores.index]['id']

    args = request.full_path
    args = unquote_plus(args)

    parameter_dict = request.args.to_dict()
    parameters = ''
    rec = ''
    for key in parameter_dict.keys():
        parameters += 'key : {}, value : {}'.format(key,request.args[key])
        if key == 'title':
            for i in range(10):
                rec += str(content_recommender(request.args[key],10).to_list()[i])
                if i<9 :
                    rec += "/" 
            
            return redirect('http://localhost:8081/recommended/rec_img/?id='+rec)


if __name__=="__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)