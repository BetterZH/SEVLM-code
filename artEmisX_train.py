import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torchvision.transforms as transforms
from transformers import GPT2Tokenizer, AutoConfig
from transformers import AdamW, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
import json
from cococaption.pycocotools.coco import COCO
from cococaption.pycocoevalcap.eval import COCOEvalCap
from accelerate import Accelerator
from models.gpt import GPT2LMHeadModel
from models.clip_vit import ImageEncoder
from models.vad_encoder import Transformer
from utils import data_utils
from utils.data_utils import *
from utils.eval_utils import top_filtering1
from utils.get_VADfeature import get_sentence_VAD
import time
import sys
import os
from utils.datasets import artEmisXTrainDataset, artEmisXEvalDataset
from PIL import Image
import tensorflow as tf
from utils.opts import parse_opt

ARTEMIS_EMOTIONS = ['amusement', 'awe', 'contentment', 'excitement',
                    'anger', 'disgust',  'fear', 'sadness', 'something else']

EMOTION_TO_IDX = {e: i for i, e in enumerate(ARTEMIS_EMOTIONS)}

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

    opt.use_cl_loss =config['use_cl_loss']
    opt.use_VAD_loss = config['use_VAD_loss']

    tokenizer = GPT2Tokenizer.from_pretrained(ckpt_path + tokenizer_name)        # load tokenizer
    decoder_model = GPT2LMHeadModel.from_pretrained(ckpt_path + model_name).to(device)   # load model with config
    ckpt_load = torch.load(ckpt_path + filename)

    if 'encoder_state_dict' in ckpt_load:
        image_encoder_state_dict = ckpt_load['encoder_state_dict']
        image_encoder.load_state_dict(image_encoder_state_dict)
        print("========= load encoder ! ")

    if 'emo_encoder_state_dict' in ckpt_load:
        opt.use_VAD = True
        emo_encoder_state_dict = ckpt_load['emo_encoder_state_dict']
        emo_encoder.load_state_dict(emo_encoder_state_dict)
        print("========= load emo_encoder ! ")
        model = nn.ModuleDict({'encoder': image_encoder, 'decoder': decoder_model, 'emo_encoder':emo_encoder})
    else:
        model = nn.ModuleDict({'encoder': image_encoder, 'decoder': decoder_model})
    optimizer = get_optimizer(model, [opt.learning_rate, opt.en_LR, opt.emo_en_LR])
    optimizer.load_state_dict(ckpt_load['optimizer_state_dict'])
    start_epoch = int(ckpt_load['epoch'])+1 
    scheduler_dic = ckpt_load['scheduler']
    best_acc = ckpt_load['best_acc']
    best_res=ckpt_load['best_res']
    del ckpt_load
    torch.cuda.empty_cache()
    print("=========== load ckpt from epoch {}, best acc is: {} , \
          best res is: {}".format(start_epoch, best_acc, best_res))

    return tokenizer, model, optimizer, scheduler_dic, start_epoch, best_acc,best_res

def save_checkpoint(name, epoch, unwrapped_model, optimizer, tokenizer, scheduler, ckpt_path, best_acc, best_res, **kwargs):
    
    model_name = 'nle_model_{}'.format(str(name))
    tokenizer_name = 'nle_gpt2_tokenizer_0'
    filename = 'ckpt_stats_' + str(name) + '.tar'
    global save_tokenizer
    if not save_tokenizer:
        tokenizer.save_pretrained(ckpt_path + tokenizer_name)   # save tokenizer
        save_tokenizer=True

    unwrapped_model.decoder.save_pretrained(ckpt_path + model_name, save_function=accelerator.save)
    if opt.fixed_encoder: 
        if opt.use_VAD:
            ckpt_load = {'epoch': epoch,
           'optimizer_state_dict': optimizer.state_dict(), 
           'emo_encoder_state_dict':unwrapped_model.emo_encoder.state_dict(),
           'scheduler': scheduler.state_dict(),
           'best_acc': best_acc,
           'best_res': best_res,
            **kwargs}
        else:
            ckpt_load = {'epoch': epoch,
           'optimizer_state_dict': optimizer.state_dict(), 
           'scheduler': scheduler.state_dict(),
           'best_acc': best_acc,
           'best_res': best_res,
            **kwargs}
        
    else:
        if opt.use_VAD:
            ckpt_load = {'epoch': epoch,
           'optimizer_state_dict': optimizer.state_dict(), 
           'scheduler': scheduler.state_dict(),
           'encoder_state_dict':unwrapped_model.encoder.state_dict(),
           'emo_encoder_state_dict':unwrapped_model.emo_encoder.state_dict(),
           'best_acc': best_acc,
           'best_res': best_res,
            **kwargs}
        else:
            ckpt_load = {'epoch': epoch,
           'optimizer_state_dict': optimizer.state_dict(), 
           'scheduler': scheduler.state_dict(),
           'encoder_state_dict':unwrapped_model.encoder.state_dict(),
           'best_acc': best_acc,
           'best_res': best_res,
            **kwargs}

    accelerator.save(ckpt_load, ckpt_path + filename)
        
def get_scores(annFile, resFile, save_scores_path,  full_predictions):
    all_file = json.load(open(opt.nle_data_val_path, 'r'))
    
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

    coco = COCO(annFile)
    cocoRes = coco.loadRes(resFile)
    cocoEval = COCOEvalCap(coco, cocoRes)
    cocoEval.evaluate()
    cocoEval.eval['sub_acc']=acc
    with open(save_scores_path, 'w') as w:
        json.dump(cocoEval.eval, w)
    
    return acc, cocoEval.eval
    
def sample_sequences(model, tokenizer, loader):
    
    model.encoder.eval()
    model.decoder.eval()
    if opt.use_VAD:
        model.emo_encoder.eval()
        
    results_exp = []
    results_full = []
    SPECIAL_TOKENS = ['<|endoftext|>', '<pad>', '<prefix>', '<emotion>', '<explanation>']
    special_tokens_ids = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
    because_token = tokenizer.convert_tokens_to_ids('Ġbecause')
    max_len = 25
    from tqdm import tqdm
    for i,batch in  tqdm (enumerate(loader),total=len(loader)):
        batch = tuple(input_tensor.to(device) for input_tensor in batch)
        img, img_id, input_ids, segment_ids, VAD_features = batch
        bt = img.size(0)
        current_outputs = torch.full([bt,1],-1).to(device)
        end_idx=[max_len]*bt
        always_exp=[False]*bt
        new_segment=torch.full([bt,1],-1).to(device)
        with torch.no_grad():
            img_embeddings = model.encoder(img)
            for step in range(max_len + 1):
                if step == max_len:
                    break
                if opt.use_VAD:
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
                            emo_features=ef_output)
                
                lm_logits = outputs.logits[0]  # [bt, len_word, vob_size]
                logits = lm_logits[:, -1, :] / temperature # [bt, vob_size]
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
                        new_segment[j] = special_tokens_ids[-1]   
               
                VAD_features = torch.stack(updataed_VAD_features).to(device).clone().detach()       
                segment_ids = torch.cat((segment_ids,new_segment),dim=1)
              
                
            for k in range(bt):
                decoded_sequences = tokenizer.decode(current_outputs[k][1:end_idx[k]], skip_special_tokens=True).lstrip()
                results_full.append({"image_id": img_id[k].item(), "caption": decoded_sequences})
                if 'because' in decoded_sequences:
                    cut_decoded_sequences = decoded_sequences.split('because')[-1].strip()
                else:
                    cut_decoded_sequences = " ".join(decoded_sequences.split()[2:])
                
                results_exp.append({"image_id": img_id[k].item(), "caption": cut_decoded_sequences})
              
    return results_full, results_exp

def get_optimizer(model, learning_rate):
    no_decay = ['bias', 'LayerNorm.weight']

    if opt.fixed_encoder:
        if opt.use_VAD:
            optimizer = AdamW([
             {'params':[p for n, p in model.emo_encoder.named_parameters() if not any(nd in n for nd in no_decay)],'weight_decay': weight_decay, 'lr':learning_rate[2]},
             {'params': [p for n, p in model.emo_encoder.named_parameters() if any(nd in n for nd in no_decay)],'weight_decay': 0.0,'lr':learning_rate[2]},
             {'params':[p for n, p in model.decoder.named_parameters() if not any(nd in n for nd in no_decay)],'weight_decay': weight_decay, 'lr':learning_rate[0]},
             {'params': [p for n, p in model.decoder.named_parameters() if any(nd in n for nd in no_decay)],'weight_decay':0.0, 'lr':learning_rate[0]}])
        else:
            optimizer_grouped_parameters_decoder = [
            {'params': [p for n, p in model.decoder.named_parameters() if not any(nd in n for nd in no_decay)],  
            'weight_decay': weight_decay},
            {'params': [p for n, p in model.decoder.named_parameters() if any(nd in n for nd in no_decay)], 
            'weight_decay': 0.0}]
            optimizer = AdamW(optimizer_grouped_parameters_decoder, lr=learning_rate[0])

    else:
        if opt.use_VAD:
            optimizer = AdamW([
             {'params':[p for n, p in model.encoder.named_parameters() if not any(nd in n for nd in no_decay)],'weight_decay': weight_decay, 'lr':learning_rate[1]},
             {'params': [p for n, p in model.encoder.named_parameters() if any(nd in n for nd in no_decay)],'weight_decay': 0.0,'lr':learning_rate[1]},
             {'params':[p for n, p in model.emo_encoder.named_parameters() if not any(nd in n for nd in no_decay)],'weight_decay': weight_decay, 'lr':learning_rate[2]},
             {'params': [p for n, p in model.emo_encoder.named_parameters() if any(nd in n for nd in no_decay)],'weight_decay': 0.0,'lr':learning_rate[2]},
             {'params':[p for n, p in model.decoder.named_parameters() if not any(nd in n for nd in no_decay)],'weight_decay': weight_decay, 'lr':learning_rate[0]},
             {'params': [p for n, p in model.decoder.named_parameters() if any(nd in n for nd in no_decay)],'weight_decay':0.0, 'lr':learning_rate[0]}])
        else:
            optimizer = AdamW([
             {'params':[p for n, p in model.encoder.named_parameters() if not any(nd in n for nd in no_decay)],'weight_decay': weight_decay, 'lr':learning_rate[1]},
             {'params': [p for n, p in model.encoder.named_parameters() if any(nd in n for nd in no_decay)],'weight_decay': 0.0,'lr':learning_rate[1]},
             {'params':[p for n, p in model.decoder.named_parameters() if not any(nd in n for nd in no_decay)],'weight_decay': weight_decay, 'lr':learning_rate[0]},
             {'params': [p for n, p in model.decoder.named_parameters() if any(nd in n for nd in no_decay)],'weight_decay':0.0, 'lr':learning_rate[0]}])
    
    return optimizer

def add_summary_value(writer, keys, value, iteration):
   with writer.as_default():
       tf.summary.scalar( keys, value,step=iteration)
       writer.flush()

accelerator = Accelerator()
device = accelerator.device
opt = parse_opt()
print(opt)

### test decoding
no_sample = True   # defalut True
top_k =  0
top_p =  0.9
temperature = 1


if (not os.path.exists(opt.ckpt_path)):
    os.makedirs(opt.ckpt_path)
    cmd='chmod -R 777 {}'.format(opt.ckpt_path)
    os.system(cmd)
if (not os.path.exists(opt.caption_save_path)):
    os.makedirs(opt.caption_save_path)
    cmd='chmod -R 777 {}'.format(opt.caption_save_path)
    os.system(cmd)

img_size = 224 # 224 for CLIP
weight_decay = 0
gradient_accumulation_steps = 1   
start_epoch = 0
save_tokenizer=False
best_acc=0.0
best_res=0.0
best_acc_epoch=0
best_res_epoch=0
no_improvment = 0
EF_DIM = 3


image_encoder = ImageEncoder(device).to(device)
encoder_out_dim=768
if opt.fixed_encoder:
    change_requires_grad(image_encoder,False)
else:
    change_requires_grad(image_encoder, True)

if opt.use_VAD:
    emo_encoder = Transformer(EF_DIM, num_layers=opt.VAD_En_Layers, nhead=opt.VAD_En_Heads, dim_feedforward = 128)
    emo_encoder = emo_encoder.to(device)

if opt.load_from_epoch is not None:
    tokenizer, model, optimizer, scheduler_dic, start_epoch, \
    best_acc, best_res = load_checkpoint(ckpt_path=opt.ckpt_path, epoch=opt.load_from_epoch)
else:
    tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
    orig_num_tokens = len(tokenizer.encoder)  
    num_new_tokens = tokenizer.add_special_tokens({'pad_token': '<pad>',
                                                    'additional_special_tokens': ['<prefix>','<emotion>','<explanation>']})
    
    assert len(tokenizer) == orig_num_tokens + num_new_tokens  
    config = AutoConfig.from_pretrained('distilgpt2')  
    # Add configs
    setattr(config, 'img_size', None)
    setattr(config, 'max_seq_len', None) 
    setattr(config,'len_prefix',None)
    setattr(config, 'prefix_size',None) 
    setattr(config,'encoder_dim', None)
    setattr(config,'use_cl_loss', None)
    setattr(config,'use_VAD_loss',None)
    config.img_size = img_size
    config.max_seq_len = opt.max_seq_len 
    config.len_prefix=opt.len_prefix
    config.add_cross_attention = True
    config.prefix_size = opt.prefix_dim
    config.encoder_dim=encoder_out_dim
    config.use_cl_loss = opt.use_cl_loss
    config.use_VAD_loss = opt.use_VAD_loss 

    decoder_model = GPT2LMHeadModel.from_pretrained('distilgpt2', config = config)
    decoder_model.resize_token_embeddings(len(tokenizer))
    decoder_model = decoder_model.to(device)
    
    if opt.use_VAD:
        model = nn.ModuleDict({'encoder': image_encoder, 'decoder': decoder_model, 'emo_encoder':emo_encoder})
    else:
        model = nn.ModuleDict({'encoder': image_encoder, 'decoder': decoder_model})
    optimizer = get_optimizer(model, [opt.learning_rate,opt.en_LR,opt.emo_en_LR])
        
print("Model Setup Ready...")

img_transform = transforms.Compose([transforms.Resize((img_size,img_size)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

train_dataset = artEmisXTrainDataset(path = opt.nle_data_train_path, 
                                 transform = img_transform, 
                                 tokenizer = tokenizer, 
                                 max_seq_len = opt.max_seq_len,
                                 len_prefix=opt.len_prefix,
                                 prompt_text=opt.prompt_text)

train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size = opt.batch_size, 
                                           shuffle=True, 
                                           pin_memory=True,
                                           num_workers=6)

val_dataset = artEmisXEvalDataset(path = opt.nle_data_val_path,      
                               transform = img_transform, 
                               tokenizer = tokenizer, 
                               max_seq_len = opt.max_seq_len,
                               len_prefix=opt.len_prefix,
                               prompt_text=opt.prompt_text)

val_loader = torch.utils.data.DataLoader(val_dataset,
                                          batch_size = opt.batch_size*2, 
                                          shuffle=False, 
                                          pin_memory=True,
                                          num_workers=6)

model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)
print("lenght of train/val set is: {}/{} ".format(len(train_dataset),len(val_dataset)))

    
num_batch=len(train_loader)
t_total = (num_batch // gradient_accumulation_steps) * opt.num_train_epochs
warmup_steps = 0 
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)

if opt.load_from_epoch is not None:
    scheduler.load_state_dict(scheduler_dic)


tf_summary_writer = tf and tf.summary.create_file_writer(opt.ckpt_path) 
for epoch in range(start_epoch, opt.num_train_epochs):
    if opt.fixed_encoder:
        model.encoder.eval()
    else:
        model.encoder.train()
    model.decoder.train()
    if opt.use_VAD:
        model.emo_encoder.train()

    accum_loss = 0
    accum_loss_xe_text = 0
    accum_loss_xe_emotion = 0
    accum_loss_exp_position = 0
    accum_loss_kl = 0
    accum_loss_cl = 0
    accum_loss_vad = 0
    for step, batch in enumerate(train_loader):
        batch = tuple(input_tensor.to(device) for input_tensor in batch)
        img, _, input_ids, labels, segment_ids,VAD_labels = batch
       
        img_shape=img.shape
        lab_shape=labels.shape
        VAD_labels_shape =VAD_labels.shape

        img=img.view(-1,img_shape[-3],img_shape[-2],img_shape[-1]) # [bt*2, 3, 224, 224]
        input_ids=input_ids.view(-1,lab_shape[-1]) # [bt*2, len]
        labels = labels.view(-1,lab_shape[-1]) # [bt*2, len]
        VAD_labels = VAD_labels.view(-1, VAD_labels_shape[-2], VAD_labels_shape[-1]) # [bt*2, len, 3]
        
        if opt.fixed_encoder:
             with torch.no_grad():
                img_embeddings = model.encoder(img) # img_embeddings:([bt*2,768],[bt*2, 196, 768])
        else:
            img_embeddings = model.encoder(img) 
        
        if opt.use_VAD:
            (_, _, ef_output) = model.emo_encoder(VAD_labels)
        else:
            ef_output = None
        
        outputs = model.decoder(input_ids=input_ids, 
                    past_key_values=None, 
                    attention_mask=None, 
                    token_type_ids=segment_ids, 
                    position_ids=None, 
                    encoder_hidden_states= img_embeddings, 
                    encoder_attention_mask=None, 
                    labels=labels, 
                    use_cache=False, 
                    return_dict=True,
                    alpha = opt.alpha,
                    cl_Loss_weight=opt.cl_Loss_weight,#
                    emo_features = ef_output, 
                    VAD_labels = VAD_labels,
                    VAD_Loss_weight=opt.VAD_Loss_weight, 
                    )
      
        loss = outputs.loss[0]
        loss = loss / gradient_accumulation_steps
        accelerator.backward(loss)
        accum_loss += loss.item()

    
        loss_xe_text = outputs.loss[1]
        if loss_xe_text is not None:
            loss_xe = loss_xe_text /gradient_accumulation_steps
            accum_loss_xe_text += loss_xe_text.item()
        
    
        loss_cl = outputs.loss[2]
        if loss_cl is not None:
            loss_cl = loss_cl /gradient_accumulation_steps
            accum_loss_cl +=loss_cl.item()

        loss_vad = outputs.loss[3]
        if loss_vad is not None:
            loss_vad = loss_vad /gradient_accumulation_steps
            accum_loss_vad += loss_vad.item()


        if step % gradient_accumulation_steps == 0 or step == len(train_loader) - 1:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            add_summary_value(tf_summary_writer, 'sum_loss', accum_loss, int(epoch*num_batch+step))
            add_summary_value(tf_summary_writer, 'loss_xe_text', accum_loss_xe_text, int(epoch*num_batch+step))
            add_summary_value(tf_summary_writer, 'loss_cl', accum_loss_cl, int(epoch*num_batch+step))
            add_summary_value(tf_summary_writer, 'loss_vad', accum_loss_vad, int(epoch*num_batch+step))
            add_summary_value(tf_summary_writer, 'lr', optimizer.state_dict()['param_groups'][0]['lr'], int(epoch*num_batch+step))
            tf_summary_writer.flush()
            accum_loss = 0  
            accum_loss_xe_text = 0
            accum_loss_cl=0
            accum_loss_vad = 0
        
       
        if  int(epoch*num_batch+(step+1)) % opt.eval_steps == 0:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model) 
            if accelerator.is_main_process:
                results_full, results_exp = sample_sequences(unwrapped_model, tokenizer, val_loader)
                resFileExp = opt.caption_save_path + 'captions_exp_' + str(epoch) + '_'+ str(step) + '_step.json'
                unf_resFileExp = opt.caption_save_path + 'unf_captions_exp_' + str(epoch) + '_'+ str(step) + '_step.json'
                unf_resFileFull = opt.caption_save_path + 'unf_captions_full_' + str(epoch) + '_'+ str(step) + '_step.json'
                save_scores_pathExp = opt.caption_save_path + 'scores_exp_' + str(epoch) + '_'+ str(step) + '_step.json' 
                with open(unf_resFileExp, 'w') as w:
                    json.dump(results_exp, w)     
                with open(unf_resFileFull, 'w') as w:
                    json.dump(results_full, w)

                acc, res = get_scores(opt.annFileExp, unf_resFileExp, save_scores_pathExp, results_full)            
                print("emotion acc is: {}".format(acc))  

                if opt.fixed_encoder:
                    model.encoder.eval()
                else:
                    model.encoder.train()
                model.decoder.train()
                if opt.use_VAD:
                    model.emo_encoder.train()

                if res == None:
                    continue
                if acc > best_acc or res['Bleu_4'] > best_res:
                    if acc > best_acc:
                        best_acc = acc
                        best_acc_epoch=epoch
                        no_improvment=0
                        save_checkpoint('best-acc', epoch, unwrapped_model, optimizer, tokenizer, scheduler, opt.ckpt_path, best_acc, best_res)
                        print(" best acc model is saved !")
                    if res['Bleu_4'] > best_res:
                        best_res = res['Bleu_4'] 
                        best_res_epoch=epoch
                        no_improvment=0
                        save_checkpoint('best-res', epoch, unwrapped_model, optimizer, tokenizer, scheduler, opt.ckpt_path, best_acc, best_res)
                        print("best result model is saved !")
                else:
                    no_improvment +=1
                    print("{} times no improvment !".format(no_improvment))
                if no_improvment > opt.stop_after_evals:
                    print("sys.exit !")
                    print("best acc is {} at epoch {}, best B4 is {} at epoch {}.".format(best_acc,best_acc_epoch,best_res,best_res_epoch) )
                    sys.exit() 
                



