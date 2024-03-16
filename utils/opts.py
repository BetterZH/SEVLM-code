# -*- coding: utf-8 -*-
import argparse


def parse_opt():
    parser = argparse.ArgumentParser()

    #### training params
    parser.add_argument('--fixed_encoder', type=bool, default=True, help='if True, fix the params of image encoder')
    parser.add_argument('--load_from_epoch',type=str, default=None, help="epoch number")
    parser.add_argument("--eval_steps",type=int, default=2000, help="evaluate model per k training steps")
    parser.add_argument("--stop_after_evals", type=int, default=10, help="If there is no increase in performance for K evaluations, the training is terminated.")
    parser.add_argument("--len_prefix", type=int, default=0,help="the length of prefix")
    parser.add_argument("--prompt_text",type=str,default=" the emotion is")
    parser.add_argument("--max_seq_len",type=int,default=30,help="the length of full explanation")
    parser.add_argument("--prefix_dim", type=int, default=768)
    parser.add_argument("--use_VAD", type=bool, default=True, help="use VAD embeddings to enhance explanation embeddings")
    parser.add_argument("--use_cl_loss", type=bool, default=True, help="use triplet CL loss")
    parser.add_argument("--use_VAD_loss", type=bool, default=True, help="use VAD loss on explanation")
    parser.add_argument("--num_train_epochs", type=int, default=35)
    parser.add_argument("--alpha", type=float, default=1.0, help="the weight of XE loss")
    parser.add_argument("--cl_Loss_weight", type=float, default=1.0)
    parser.add_argument("--VAD_Loss_weight",type=float,default=1.0)
    parser.add_argument("--learning_rate",type=float,default=2e-5, help="the LR of decoder")
    parser.add_argument("--en_LR",type=float,default=2e-5, help="the LR of image encoder")
    parser.add_argument("--emo_en_LR",type=float,default=4e-5, help="the LR of VAD encoder")
    parser.add_argument("--batch_size",type=int,default=32)
    parser.add_argument("--VAD_En_Layers",type=int,default=3)
    parser.add_argument("--VAD_En_Heads",type=int,default=1)

    ### file path
    parser.add_argument("--ckpt_path",type=str,default='/xxxx/ckpts/')
    parser.add_argument("--caption_save_path",type=str,default="/xxxx/results/")
    parser.add_argument("--annFileExp",type=str,default='cococaption/annotations/artEmisX_val_annot_exp.json')
    parser.add_argument("--nle_data_train_path",type=str,default='/xxxx/data/artEmis/artEmisX_cl_train.json')
    parser.add_argument("--nle_data_val_path",type=str,default='/xxxx/data/artEmis/artEmisX_val.json')
    
    args = parser.parse_args()
    
    return args
