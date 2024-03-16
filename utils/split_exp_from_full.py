import json


with open('/xxxxx/results/xxxxx_full.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

for item in data:
    if 'caption' in item:
        index_because = item['caption'].find('because')
        if index_because != -1:
            new_caption = item['caption'][index_because + len('because'):].strip()
            item['caption'] = new_caption
        else:
            item['caption'] = 'null'
        if item['caption']=="":
            item['caption'] = 'null'

with open('xxxxxx/results/xxxxx_exp.json', 'w', encoding='utf-8') as file:
    json.dump(data, file, indent=2, ensure_ascii=False)

