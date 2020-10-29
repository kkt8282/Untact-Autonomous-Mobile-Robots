pos_dict = {
    'Adverb': {'건강검진', '예약', '유의', '주의', '진료절차', 
               '접수', '예약시간', '입원', '절차', '퇴원'}
}
from soynlp.postagger import Dictionary
from soynlp.postagger import LRTemplateMatcher
from soynlp.postagger import LREvaluator
from soynlp.postagger import SimpleTagger
from soynlp.postagger import UnknowLRPostprocessor
tagger = SimpleTagger(LRTemplateMatcher(Dictionary(pos_dict)), LREvaluator(), UnknowLRPostprocessor())

def sequence(sent):
    token_list = tagger.tag(sent)
    list_word = []
    for ele in token_list:
        if ele[1] == 'Adverb':
            word, adverb = ele
            list_word.append(word)
        
        
    if '건강검진' in list_word:
        if '예약' in list_word or '접수' in list_word:
            return "054에 260에 8188로 전화하면 돼요"
        elif '유의' or '주의' in list_word:
            return "건강검진 후에는 술과 담배, 매운 음식을 자제하고 수면 내시경 고객은 당일 운전을 금하셔야 해요"
        elif '진료' or '절차' in list_word:
            return "예약, 접수, 탈의, 검사진행, 검사종료 후 결과 상담으로 진행 돼요"
        else:
            return "조금만 풀어서 설명해주시겠어요"
    if '접수' in list_word: 
        if '접수' or '예약' or '절차' in list_word:
            return "인터넷 접수하거나 054-260-8001로 전화 주세요"
        elif '예약시간' in list_word:
            return "검진예약시간은 평일 오전 8시 부터 오후 5시까지이고 토요일엔 오전 8시부터 정오 12시까지에요"
        elif '유의' or '주의' in list_word:
            return "예약 시간을 지키고 금식을 유지해주세요"
        else:
            return "조금만 풀어서 설명해주시겠어요"
    elif '입원' in list_word:
        if '입원' or '절차' in list_word:
            return "입원 약정서를 작성한 후 입원 결정서가 제출 되면 입실이 진행돼요"
        else:
            return "조금만 풀어서 설명해주시겠어요"
    elif '퇴원' in list_word:
        if '퇴원' or '절차' in list_word:
            return '퇴원 수속 안내문과 입원 진료비를 납부 한 후 간호사의 안내에 따라 귀가 하시면 돼요'
        else:
            return "조금만 풀어서 설명해주시겠어요" 

    else:
        return "조금만 풀어서 설명해주시겠어요"
