import json
from .get_wordBase import lemmatize_sentence
def get_VAD_dict(file_path):
    vad_dict = {}
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split('\t')
            if len(parts) == 4:
                word = parts[0]
                vad_values = [float(part) for part in parts[1:]]
                vad_dict[word] = vad_values
    return vad_dict

vad_dict_path = '/xxx/pathto/NRC-VAD-Lexicon/BipolarScale/NRC-VAD-Lexicon.txt' 
VAD_dict = get_VAD_dict(vad_dict_path)

def get_sentence_VAD(sentence):
    words_VAD_values=[]
    baseForm_tokens = lemmatize_sentence(sentence)
    for word in baseForm_tokens:
        if word in VAD_dict:
            words_VAD_values.append(VAD_dict[word])
        else:
            words_VAD_values.append([0.,0.,0.])
    return words_VAD_values