import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from transformers import GPT2Tokenizer, AutoConfig
from transformers import AdamW, get_linear_schedule_with_warmup
import json
from cococaption.pycocotools.coco import COCO
from cococaption.pycocoevalcap.eval import COCOEvalCap
from PIL import Image
from accelerate import Accelerator
from models.gpt import GPT2LMHeadModel
from models.clip_vit import ImageEncoder
from utils import data_utils
from utils.data_utils import *
from utils.eval_utils import top_filtering1
import time
import pandas as pd
from basics import pickle_data, unpickle_data, torch_load_model,Vocabulary
import itertools
from single_caption_per_image import apply_basic_evaluations
import os
import numpy as np
import tqdm
from models.resnet_encoder import ResnetEncoder
from models.vad_encoder import Transformer
from utils.get_VADfeature import get_sentence_VAD

ARTEMIS_EMOTIONS = ['amusement', 'awe', 'contentment', 'excitement',
                    'anger', 'disgust',  'fear', 'sadness', 'something else']

EMOTION_TO_IDX = {e: i for i, e in enumerate(ARTEMIS_EMOTIONS)}
import unicodedata


def unicode_escaping(s):
    result = ""
    for c in s:
        if unicodedata.category(c) in ("Mn", "Mc"):
            result += f"#U{ord(c):04x}"
        else:
            result += c
    return result


def change_requires_grad(model, req_grad):
    for p in model.parameters():
        p.requires_grad = req_grad


def load_checkpoint(ckpt_path, epoch):
    
    model_name = 'nle_model_{}'.format(str(epoch))
    tokenizer_name = 'nle_gpt2_tokenizer_0'
    filename = 'ckpt_stats_' + str(epoch) + '.tar'
    image_encoder_state_dict =None

    file_config=ckpt_path+str(model_name)+'/'+'config.json'
    config = json.load(open(file_config,'r'))
  
    global use_cl_loss
    global use_VAD_loss
    global use_VAD

    use_cl_loss = config['use_cl_loss'] if 'use_cl_loss' in config.keys() else None
    use_VAD_loss = config['use_VAD_loss'] if 'use_VAD_loss' in config.keys() else None

    tokenizer = GPT2Tokenizer.from_pretrained(ckpt_path + tokenizer_name)        # load tokenizer
    decoder_model = GPT2LMHeadModel.from_pretrained(ckpt_path + model_name).to(device)   # load model with config
    image_encoder = ImageEncoder(device).to(device)
    change_requires_grad(image_encoder,False)
    opt = torch.load(ckpt_path + filename)
    if 'encoder_state_dict' in opt:
        image_encoder_state_dict = opt['encoder_state_dict']
        image_encoder.load_state_dict(image_encoder_state_dict)
        print("load encoder!")
    if 'emo_encoder_state_dict' in opt:
        use_VAD = True
        emo_encoder = Transformer(EF_DIM, num_layers=3, nhead=1, dim_feedforward = 128)
        emo_encoder = emo_encoder.to(device)
        emo_encoder_state_dict = opt['emo_encoder_state_dict']
        emo_encoder.load_state_dict(emo_encoder_state_dict)
        print(" load emo_encoder ! ")
        model = nn.ModuleDict({'encoder': image_encoder, 'decoder': decoder_model, 'emo_encoder':emo_encoder})
    else:
        model = nn.ModuleDict({'encoder': image_encoder, 'decoder': decoder_model})

    start_epoch = int(opt['epoch'])+1 
    
    del opt
    torch.cuda.empty_cache()
    print("load ckpt from epoch {} ".format(start_epoch))

    return tokenizer, model, start_epoch

        
def get_scores(annFile, resFile, save_scores_path,  full_predictions,save_scores_pathExp_details):
    all_file = json.load(open(nle_data_test_path, 'r'))
    
    gt_answers = {}
    for key,value in all_file.items():
        has_max,maximizer=data_utils.proc_distribution(value['origin_emotion_distribution'])
        if has_max:
            gt_answers[int(key)]= maximizer

    pred_answers = {}
    for item in full_predictions:
        if gt_answers.get(item['image_id']) is not None:
            pred_answers[item['image_id']] = item['caption'].split("because")[0].strip()
          
 
    correct_keys = []
    for key,value in pred_answers.items():
        gt_answer = gt_answers[key]
        if value == gt_answer: 
            correct_keys.append(key)
   
    if len(correct_keys) ==0 :
        return -1,None 

    acc = len(correct_keys)/len(gt_answers) 
    print("acc is {}".format(acc))

    coco = COCO(annFile)
    cocoRes = coco.loadRes(resFile)
    cocoEval = COCOEvalCap(coco, cocoRes)
    cocoEval.evaluate()
    cocoEval.eval['sub_acc']=acc
    cocoEval.imgToEval
    with open(save_scores_path, 'w') as w:
        json.dump(cocoEval.eval, w)
    return acc, cocoEval.eval



class artEmisXEvalDataset(Dataset):

    def __init__(self, path, transform, tokenizer, max_seq_len,len_prefix=0):

        self.tokenizer = tokenizer
        self.transform = transform
        self.max_seq_len = max_seq_len       #
        self.data = json.load(open(path, 'r'))
        self.ids_list = list(self.data.keys())
        self.len_prefix=len_prefix
        self.n_emotions=9


    def __getitem__(self, i):
        
        data_id = self.ids_list[i]
        sample = self.data[data_id]
        img_name = sample['image_name']  
        tep = img_name.split('/')
        art_style=tep[0]
        painting = tep[-1][:-len('.jpg')] 
        # tokenization process
        pf_segment_id, emo_segment_id, exp_segment_id = self.tokenizer.convert_tokens_to_ids(['<prefix>', '<emotion>', '<explanation>'])
        
        pf_token =  '<prefix>'                                                          
        tokens= [pf_token] * self.len_prefix                                                                                                                                      
        segment_ids = [pf_segment_id] * self.len_prefix

        emotion = [self.tokenizer.bos_token] + self.tokenizer.tokenize(prompt_text)
        emotion_len = len(emotion )
        tokens += emotion 

        segment_ids += [emo_segment_id] * emotion_len

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        segment_ids = torch.tensor(segment_ids, dtype=torch.long)
        ## get VAD for tokens
        words = [token.replace('Ġ', '') for token in tokens]
        VAD_features = get_sentence_VAD(words)
        VAD_features = torch.tensor(VAD_features, dtype=torch.float)

        folder = '/xxxx/wikiart_rescaled_max_size_to_600px_same_aspect_ratio/'
        img_path = folder + img_name

        try:
            img = Image.open(img_path)
        except:
            nfc= unicodedata.normalize('NFC', img_path)
            img_path = unicode_escaping(nfc)
            img = Image.open(str(img_path.encode('unicode_escape').decode()).replace('\\x','#U00'))

        if img.mode is not 'RGB':
            img = img.convert('RGB')
        img = self.transform(img)
        did = torch.LongTensor([int(data_id)])
        
        return (img, did, input_ids, segment_ids, art_style, painting, VAD_features,img_name)

    def __len__(self):
        return len(self.ids_list)


def sample_sequences(model, tokenizer, loader):
    
    model.encoder.eval()
    model.decoder.eval()
    if use_VAD:
        model.emo_encoder.eval()

    results_exp = []
    results_full = []
    final_results = []
    tsne_results = []

    column_names = ['art_style', 'painting', 'grounding_emotion', 'caption']
    df=pd.DataFrame(columns=column_names)
    SPECIAL_TOKENS = ['<|endoftext|>', '<pad>', '<prefix>', '<emotion>', '<explanation>']
    special_tokens_ids = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
    because_token = tokenizer.convert_tokens_to_ids('Ġbecause')
    max_len = 25

    for batch in tqdm.tqdm(loader):
        img, img_id, input_ids, segment_ids, art_style, painting,VAD_features,img_name = batch
        img = img.to(device)
        img_id = img_id.to(device)
        input_ids = input_ids.to(device)
        segment_ids = segment_ids.to(device)
        
        bt = img.size(0)
        current_outputs = torch.full([bt,1],-1).to(device)
        end_idx=[max_len]*bt
        always_exp=[False]*bt
        new_segment=torch.full([bt,1],-1).to(device)
        with torch.no_grad():
            for step in range(max_len + 1):
                img_embeddings = model.encoder(img)
                if step == max_len:
                    break

                if use_VAD:
                    (_, _, ef_output) = model.emo_encoder(VAD_features)        
                else:
                    ef_output = None

                
                outputs = model.decoder(input_ids=input_ids, 
                            past_key_values=None, 
                            attention_mask=None, 
                            token_type_ids=segment_ids, 
                            position_ids=None, 
                            encoder_hidden_states=img_embeddings, 
                            encoder_attention_mask=None, 
                            labels=None,
                            use_cache=False, 
                            return_dict=True,
                            emo_features=ef_output,)
              
                lm_logits = outputs.logits[0] 
                logits = lm_logits[:, -1, :] / temperature 
                logits = top_filtering1(logits, top_k=top_k, top_p=top_p)  
                probs = F.softmax(logits, dim=-1)
                prev = torch.topk(probs, 1)[1] if no_sample else torch.multinomial(probs, 1) 
                updataed_VAD_features = []
                input_ids = torch.cat((input_ids, prev), dim = 1)
                current_outputs  = torch.cat((current_outputs, prev), dim = 1)

                for j in range(bt):
                    if prev[j].item() in special_tokens_ids and end_idx[j]==max_len:
                        end_idx[j]=step+1

                    decoded_sequences = [tokenizer.decode(wj, skip_special_tokens=False).lstrip() for wj in input_ids[j]]
                    VAD_values= get_sentence_VAD(decoded_sequences)
                    updataed_VAD_features.append(torch.tensor(VAD_values,dtype=torch.float))

                    if not always_exp[j]:
                        if prev[j].item() != because_token:
                            new_segment[j] = special_tokens_ids[-2]   
                        else:
                            new_segment[j] = special_tokens_ids[-1]   
                            always_exp[j] = True
                    else:
                        new_segment[j] = special_tokens_ids[-1]   # explanation segment
                        
                VAD_features = torch.stack(updataed_VAD_features).to(device).clone().detach()  
                segment_ids = torch.cat((segment_ids,new_segment),dim=1)
                
              
            for k in range(bt):
                decoded_sequences = tokenizer.decode(current_outputs[k][1:end_idx[k]], skip_special_tokens=True).lstrip()
                results_full.append({"path":str(art_style[k]+'/'+painting[k]),"image_id": img_id[k].item(), "caption": decoded_sequences})
                if 'because' in decoded_sequences:
                    cut_decoded_sequences = decoded_sequences.split('because')[-1].strip()
                else:
                    cut_decoded_sequences = " ".join(decoded_sequences.split()[2:])
                if len(cut_decoded_sequences)==0:
                    cut_decoded_sequences='<unk>'

                emotion = decoded_sequences.split("because")[0].strip()
                df_tem = pd.DataFrame({'art_style':[art_style[k]], 'painting':[painting[k]],'grounding_emotion':[emotion], 'caption': [cut_decoded_sequences]})
                df=df.append(df_tem,ignore_index=True)
                results_exp.append({"image_id": img_id[k].item(), "caption": cut_decoded_sequences})
                tsne_results.append({"image_id": img_id[k].item(),'img_name':img_name[k], "preEmotion":emotion,"explanation":cut_decoded_sequences, "image_fts":(outputs.logits[2][k]).cpu().numpy(),"explanation_fts":(outputs.logits[3][k]).cpu().numpy()})
    
    final_results.append(df)
    pickle_data(out_file, final_results)
    print('pickle data Done')       
    return results_full, results_exp,tsne_results


def get_optimizer(model, learning_rate):
    no_decay = ['bias', 'LayerNorm.weight']

    optimizer = AdamW([
             {'params':[p for n, p in model.encoder.named_parameters() if not any(nd in n for nd in no_decay)],'weight_decay': 0.0, 'lr':learning_rate[1]},
             {'params': [p for n, p in model.encoder.named_parameters() if any(nd in n for nd in no_decay)],'weight_decay': 0.0,'lr':learning_rate[1]},
             {'params':[p for n, p in model.decoder.named_parameters() if not any(nd in n for nd in no_decay)],'weight_decay': 0.0, 'lr':learning_rate[0]},
             {'params': [p for n, p in model.decoder.named_parameters() if any(nd in n for nd in no_decay)],'weight_decay':0.0, 'lr':learning_rate[0]}])
    
    return optimizer


accelerator = Accelerator()
device = accelerator.device
no_sample = True # default is True   
top_k =  0
top_p =  0.9
EF_DIM = 3
load_from_epoch = "best-res"  # best-acc
model_path='xxxxxxx'
prompt_text=" the emotion is"
len_prefix=0 # default is 0
max_seq_len = 30 
use_exp_loss =None # default is None; will be updated in load function
use_VAD_loss = None  # default is None; will be updated in load function
use_VAD =  False #  default is False; will be updated in load function
img_size = 224 
ckpt_path = '/xxxx/ckpts/{}/'.format(model_path)
caption_save_path = '/xxxx/results/{}/'.format(model_path)
out_file = '/xxxx/results/{}/full_{}.pkl'.format(model_path, load_from_epoch)
out_metrics = '/xxxx/results/{}/emo_metrics__{}.csv'.format(model_path, load_from_epoch)
 

if (not os.path.exists(caption_save_path)):
    os.mkdir(caption_save_path)
    cmd='chmod -R 777 {}'.format(caption_save_path)
    os.system(cmd)

annFileExp = 'cococaption/annotations/artEmisX_test_annot_exp.json'
nle_data_test_path = '/xxxx/data/artEmis/artEmisX_test.json'
batch_size = 64 # per GPU 
start_epoch = 0
temperature = 1

if load_from_epoch is not None:
    tokenizer, model, start_epoch= load_checkpoint(ckpt_path, load_from_epoch)
    print("Model Setup Ready...")
    optimizer = get_optimizer(model, [2e-5,2e-5])



img_transform = transforms.Compose([transforms.Resize((img_size,img_size)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


test_dataset = artEmisXEvalDataset(path = nle_data_test_path,      
                               transform = img_transform, 
                               tokenizer = tokenizer, 
                               max_seq_len = max_seq_len,
                               len_prefix=len_prefix)

test_loader = torch.utils.data.DataLoader(test_dataset,
                                          batch_size = batch_size, 
                                          shuffle=False, 
                                          pin_memory=True,
                                          num_workers=6)


if load_from_epoch is not None:
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model) 
    model, optimizer, test_loader = accelerator.prepare(model, optimizer, test_loader)

if accelerator.is_main_process:
    results_full, results_exp, tsne_results = sample_sequences(unwrapped_model, tokenizer, test_loader)
    resFileExp = caption_save_path + 'captions_exp_{}.json'.format(load_from_epoch)
    unf_resFileExp = caption_save_path + 'unf_captions_exp_{}.json'.format(load_from_epoch)
    unf_resFileFull = caption_save_path + 'unf_captions_full_{}.json'.format(load_from_epoch)
    save_scores_pathExp = caption_save_path + 'scores_exp_{}.json'.format(load_from_epoch)
    save_scores_pathExp_details = caption_save_path + 'scores_details_exp_{}.json'.format(load_from_epoch)

    with open(unf_resFileExp, 'w') as w:
        json.dump(results_exp, w)
        
    with open(unf_resFileFull, 'w') as w:
        json.dump(results_full, w)

    # acc and unfiltered results
    acc, res = get_scores(annFileExp, unf_resFileExp, save_scores_pathExp, results_full,save_scores_pathExp_details)   


references_file = '/xxxx/data/artemis/preprocess_data_nets/artemis_gt_references_grouped.pkl'
#"/xxxxx/data/artemis-v2/preprocess_data_combined/train/artemis_gt_references_grouped.pkl"
#'/xxxx/data/artemis/preprocess_data_nets/artemis_gt_references_grouped.pkl'

text2emo_path = '/xxxx/data/artemis/preprocess_data_nets/txt_to_emotion/lstm_based/best_model.pt'
#'/xxxx/data/artemis-V2/preprocess_data_combined/txt_to_emotion/lstm_based/best_model.pt'
#'/xxxx/data/artemis/preprocess_data_nets/txt_to_emotion/lstm_based/best_model.pt'

vocab_path =  '/xxxx/data/artemis/preprocess_data_nets/vocabulary.pkl'
# "/xxxx/data/artemis-V2/preprocess_data_combined/train/vocabulary.pkl"
# '/xxxx/data/artemis/preprocess_data_nets/vocabulary.pkl'

sampled_captions_file = out_file 
split = 'test' 
gpu_id = 0


#
# Load grouped GT references, Vocab & image2emotion net (for Emo-alignment)
#
gt_data = next(unpickle_data(references_file))
mask = gt_data['train']['emotion'].apply(set).apply(len)>1
train_utters  = gt_data['train']['references_pre_vocab'][mask]
train_utters = list(itertools.chain(*train_utters))  # undo the grouping per artwork to a single large list
print('Training Utterances', len(train_utters))
unique_train_utters = set(train_utters)
print('Unique Training Utterances', len(unique_train_utters))
    
# now focus on the data (split) that you created captions for
gt_data = gt_data[split]
print('Images Captioned', len(gt_data))


device = torch.device("cuda:" + str(0))
txt2emo_clf = torch_load_model(text2emo_path, map_location=device)


txt2emo_vocab = Vocabulary.load(vocab_path)
print('vocab size', len(txt2emo_vocab))

evaluation_methods =  {'emo_alignment', 'metaphor'} 

def print_out_some_basic_stats(captions):
    """ Helper function -- to print basic statistics of sampled generations
    Input: captions dataframe with column names caption
    """
    print('Some basic statistics:')
    mean_length = captions.caption.apply(lambda x: len(x.split())).mean()
    print(f'average length of productions {mean_length:.4}')
    unique_productions = len(captions.caption.unique()) / len(captions)
    print(f'percent of distinct productions {unique_productions:.4}')
    maximizer = captions.caption.mode()  
    print(f'Most common production "{maximizer.iloc[0]}"')
    n_max = sum(captions.caption == maximizer.iloc[0]) 
    print(f'Most common production appears {n_max} times -- {n_max/ len(captions):.4} frequency.')
    u_tokens = set()
    captions.caption.apply(lambda x: [u_tokens.add(i) for i in x.split()]);
    print(f'Number of distinct tokens {len(u_tokens)}')
    return unique_productions

saved_samples = next(unpickle_data(sampled_captions_file))

for  captions in saved_samples:  # you might have sampled under several sampling configurations      
    print('\nSome Random Samples:')    
    rs = captions.sample(min(len(captions), 5))[['caption', 'grounding_emotion']]        
    for _, row in rs.iterrows():
        if row.grounding_emotion is not None:
            print(row.grounding_emotion.capitalize(), end=' --- ')
        print(row.caption)
    
    print()        
    unique_productions=print_out_some_basic_stats(captions)
    print()
    

    merged = pd.merge(gt_data, captions)  # this ensures proper order of captions to gt (via accessing merged.captions)
    hypothesis = merged.caption
    references = merged.references_pre_vocab # i.e., use references that do not have <UNK>
    ref_emotions = merged.emotion
    metrics_eval = apply_basic_evaluations(hypothesis, references, ref_emotions, txt2emo_clf, txt2emo_vocab, 
                                           train_utterances=unique_train_utters,
                                           methods_to_do=evaluation_methods)

    stats = pd.Series(unique_productions, dtype=float)
    stat_track = ['mean', 'std']
    stats = stats.describe()[stat_track]
    stats = pd.concat([pd.Series({'metric': 'unique_productions'}), stats])
    metrics_eval.append(stats)
    df_metrics=pd.DataFrame(metrics_eval)
    print(df_metrics)
    print()
    df_metrics.to_csv(out_metrics)
