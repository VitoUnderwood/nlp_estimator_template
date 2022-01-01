# coding=utf-8
import re


def merge_once(sentence, max_ngram_length=4):
    """ 合并n-gram相邻的相同字符 """
    final_merge_sent = sentence
    max_ngram_length = min(max_ngram_length, len(sentence))
    for i in range(max_ngram_length, 0, -1):
        start = 0
        end = len(final_merge_sent) - i + 1
        ngrams = []
        while start < end:
            ngrams.append(final_merge_sent[start: start + i])
            start += 1
        result = []
        for cur_word in ngrams:
            result.append(cur_word)
            if len(result) > i:
                pre_word = result[len(result) - i - 1]
                if pre_word == cur_word:
                    for k in range(i):
                        result.pop()

        cur_merge_sent = ""
        for word in result:
            if not cur_merge_sent:
                cur_merge_sent += word
            else:
                cur_merge_sent += word[-1]
        final_merge_sent = cur_merge_sent
    return final_merge_sent


def merge(sentence, max_ngram_length=4):
    new_sent = merge_once(sentence, max_ngram_length)
    while len(new_sent) != len(sentence):
        sentence = new_sent
        new_sent = merge_once(sentence, max_ngram_length)
    return new_sent


def merge_sent(text):
    pattern = r',|\.|/|;|\'|`|\[|\]|<|>|\?|:|"|\{|\}|\~|!|@|#|\$|%|\^|&|\(|\)|-|=|\_|\+|，|。|、|；|‘|’|【|】|·|！| |…|（|）'
    result_list = re.split(pattern, text)
    new_result_list = list(set(result_list))
    new_result_list.sort(key=result_list.index)
    # print(result_list)
    # print(new_result_list)
    return "，".join(new_result_list)

# s = merge_sent("这款衬衫采用了经典的圆领设计，修饰颈部线条的同时，彰显出女性的干练的气质。又不失优雅气质。彰显出女性的优雅气质。彰显出女性的优雅气质。展现女性的优雅气质。")
# print(s)
