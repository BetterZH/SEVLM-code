# Training A Small Emotional Vision Language Model for Visual Art Comprehension
Official Code for SEVLM:  <br>
[arXiv](https://arxiv.org/xxxx) 
<br>


### Requirements
- [PyTorch](https://pytorch.org/) 1.8 or higher
- `pip install git+https://github.com/openai/CLIP.git`
-  `pip install transformers`
-  `pip install git+https://github.com/huggingface/accelerate`

### Images Download
We conduct experiments on artEmis Dataset [WikiArt dataset](https://github.com/cs-chan/ArtGAN/tree/master/WikiArt%20Dataset), and then resize images into a 600px resized folder, named `wikiart_rescaled_max_size_to_600px_same_aspect_ratio` 

### Data preprocessing
We conduct experiments on two benchmark datasets: [ArtEmis v1.0 ](https://arxiv.org/abs/2101.07396)  and [ArtEmis v2.0](https://arxiv.org/abs/2204.07660). Taking the data set v1.0 as an example, we introduce the following data preprocessing steps.
#### step 1: 
- download the [dataset](https://github.com/optas/artemis?tab=readme-ov-file) associated with ArtEmis v1.0.
#### step 2: 
- Follow the [ArtEmis](https://github.com/optas/artemis?tab=readme-ov-file) and store the processing results into two folder, named `preprocess_data_mini` and `preprocess_data_nets`. The `image-emotion-histogram.csv` in the former will be used in step 3. The latter mainly contains four files:  `artemis_preprocessed.csv` requires further processing in step 3, and `artemis_gt_references_grouped.pkl`, `vocabulary.pkl` and `best_model.pt` are used for evaluation.
#### step 3: 
- Merge file `preprocess_data_mini\image-emotion-histogram.csv` into file `preprocess_data_nets\artemis_preprocessed.csv` and rename column `emotion_histogram` to `origin_emotion_distribution`.
- Filter to data with repetition greater than 40 in `preprocess_data_nets\artemis_preprocessed.csv`.
- According to train/val/test splits, `preprocess_data_nets\artemis_preprocessed.csv` is divided into three json files: `artEmisX_train.json`, `artEmisX_test.json`, and `artEmisX_val.json`.
- For using contrastive head, you need to select images where the number of different emotions is greater than 1 from `artEmisX_train.json`, and save  them in `artEmisX_cl_train.json`.
- The data format of the json file is as follows: (`"1763323660598682939"` is image_id.)
```
{"1763323660598682939": 
      {"emotions": ["contentment", "awe", "something else"], 
      "explanations": ["the light grey looks like the exhaust of two rockets taking off for space", "the textures are very appealing to look at", "this reminds me of more edgy modern art kinda dark and cold but has some meaning to each person interesting"], 
      "image_name": "Abstract_Expressionism/aaron-siskind_chicago-1951.jpg",  "origin_emotion_distribution": [0.0, 0.3333333333333333, 0.3333333333333333, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3333333333333333]}, 
      ...,
"8477878696416252642": 
      {"emotions": ["contentment", "excitement", "contentment"], 
      "explanations": ["the colors go together nicely", "the bold colors make this picture come to life", "there is not much to the painting and the shapes are simple"], 
      "image_name": "Abstract_Expressionism/esteban-vicente_blue-red-black-and-white-1961.jpg", "origin_emotion_distribution": [0.0, 0.0, 0.6666666666666666, 0.3333333333333333, 0.0, 0.0, 0.0, 0.0, 0.0]},
      ...
      ,
}
```

The data structure under the folder is as follows:
```
├─artEmis
  │
  │──artEmisX_cl_train.json
  │──artEmisX_test.json
  │──artEmisX_val.json
  └──preprocess_data_mini
        │──image-emotion-histogram.csv
  └──preprocess_data_nets
        │──artemis_preprocessed.csv
        │──artemis_gt_references_grouped.pkl
        │──vocabulary.pkl  
        └──txt_to_emotion
               └──lstm_based
                      └──best_model.pt
```




#### step 4: 
You also need create the annotation `artEmisX_test_annot_exp.json` in the correct format and use[cococaption](https://github.com/tylin/coco-caption) in order to perform evaluation.


### Code
Please run from the command line with: <br>
```bash
accelerate launch artEmisX_train.py --ckpt_path /xxxx/ckpts/  --caption_save_path /xxxx/results/  --nle_data_train_path /xxxx/data/artEmis/artEmisX_cl_train.json  --nle_data_val_path /xxxx/data/artEmis/artEmisX_val.json
```

### Citation
If you find this work useful in your research, please consider citing:
```

```

## Acknowledgement

- We thank [NLX-GPT](https://github.com/fawazsammani/nlxgpt) for  open-source implementation of their language model. SEVLM repo is built on [NLX-GPT](https://github.com/fawazsammani/nlxgpt).
