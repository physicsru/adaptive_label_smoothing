import torch
from sentence_transformers import SentenceTransformer, SentencesDataset, InputExample,  LoggingHandler, losses, evaluation
from torch.utils.data import DataLoader
import json
device = torch.device("cuda")
#torch.cuda.set_device(5)
model = SentenceTransformer('bert-large-nli-stsb-mean-tokens',device = device)
train_examples = []
f = open('/home/w00536717/hammer/DSCMR-master/data/dataset_flickr30k.json')
load_dict = json.load(f)
for i in range(len(load_dict['images'])):
    for j in range(5):
        for k in range(j + 1, 5):
            t1 = load_dict['images'][i]['sentences'][j]['raw'].lower()
            t2 = load_dict['images'][i]['sentences'][k]['raw'].lower()
            train_examples.append(InputExample(texts=[t1, t2], label=1))
        for k in range(i + 1, len(load_dict['images'])):
            for kk in range(5):
                t1 = load_dict['images'][i]['sentences'][j]['raw'].lower()
                t2 = load_dict['images'][k]['sentences'][kk]['raw'].lower()
                train_examples.append(InputExample(texts=[t1, t2], label=0))
train_examples = []
f = open('/home/w00536717/hammer/DSCMR-master/data/dataset_flickr30k.json')
load_dict = json.load(f)
for i in range(len(load_dict['images'])):
    print(i)
    for j in range(5):
        for k in range(j + 1, 5):
            t1 = load_dict['images'][i]['sentences'][j]['raw'].lower()
            t2 = load_dict['images'][i]['sentences'][k]['raw'].lower()
            train_examples.append(InputExample(texts=[t1, t2], label=1))
#     for k in range(i + 1, len(load_dict['images'])):
#         t1 = load_dict['images'][i]['sentences'][0]['raw'].lower()
#         t2 = load_dict['images'][k]['sentences'][0]['raw'].lower()
#         train_examples.append(InputExample(texts=[t1, t2], label=0))

train_dataset = SentencesDataset(train_examples, model)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=64)
train_loss = losses.ContrastiveLoss(model=model)
#Tune the model
sentences1 = ['This list contains the first column', 'With your sentences', 'You want your model to evaluate on']
sentences2 = ['Sentences contains the other column', 'The evaluator matches sentences1[i] with sentences2[i]', 'Compute the cosine similarity and compares it to scores[i]']
scores = [0.3, 0.6, 0.2]
evaluator = evaluation.EmbeddingSimilarityEvaluator(sentences1, sentences2, scores)
model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=10, warmup_steps=100, output_path="./flickerbertmodel/", evaluator=evaluator, evaluation_steps=5)
