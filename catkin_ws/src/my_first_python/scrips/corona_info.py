from urllib.request import urlopen
from bs4 import BeautifulSoup
import re
from soynlp.postagger import Dictionary
from soynlp.postagger import LRTemplateMatcher
from soynlp.postagger import LREvaluator
from soynlp.postagger import SimpleTagger
from soynlp.postagger import UnknowLRPostprocessor

html = urlopen("https://terms.naver.com/entry.nhn?docId=5912275&cid=43667&categoryId=43667")
bsObj = BeautifulSoup(html, "html.parser")
corona = bsObj.findAll("div", {"class" : "se_cellArea"})
pos_dict = {
    'Adverb': {'최초발생', '근원지', '처음', '전파경로', '경로', '전염경로','접촉', '잠복기', '잠복기간', '잠복일', '증상', '감염', '행동', '치명률', '치사율', '사망률',},
}

tagger = SimpleTagger(LRTemplateMatcher(Dictionary(pos_dict)), LREvaluator(), UnknowLRPostprocessor())

def corona_info(sent):
    word = ''
    token_list = tagger.tag(sent)
    for ele in token_list:
        if ele[1] == 'Adverb':
            word, adverb = ele

    # 최초발생
    occur = []
    for i in list(corona[3].text.strip()):
        if i == '(' or i == ')' or i == '湖' or i == '北' or i == '武' or i == '漢':
            continue
        occur.append(i)
    occur = ''.join(occur)
    #감염경로
    path = []
    for i in list(corona[9].text.strip()):
        if i == '•':
            continue
        path.append(i)
    path = ''.join(path)
    path = path[25:43]+"으로 전달되며, "+ path[57:78]+" 만지면 전파됩니다."
    #잠복기
    latent = []
    for i in corona[11].text.strip():
        latent.append(i)
        if i == '(':
            break
    latent = ''.join(latent)
    latent = latent.replace('~', '일에서 ')
    latent = latent.replace('(', '입니다 ')
    #증상
    symptom = corona[13].text[:49]+"납니다."+corona[13].text[51:85]+"납니다."
    #치명률
    critical = []
    for i in corona[17].text.strip():
        critical.append(i)
        if i == '(':
            break
    critical.remove('•')
    critical.remove('(')
    critical = ''.join(critical)
    critical = critical.replace('%', '퍼센트 입니다.')
    #룰베이스
    if word == '최초발생' or word == '근원지' or word == '처음':
        return occur + "입니다"
    elif word == '전파경로' or word == '전염경로' or word == '접촉' or word == '경로':
        return path
    elif word == '잠복기' or word == '잠복기간' or word == '잠복일':
        return latent
    elif word == '증상' or word == '감염' or word == '행동':
        return symptom
    elif word == '치명률' or word == '치사율' or word == '사망률':
        return critical
    else:
        return '잘못 인식하였습니다 다르게 질문하여 주세요'
