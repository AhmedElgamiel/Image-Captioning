import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1 ,  drop_prob=0.2):
        super().__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        
        self.caption_embeddings = nn.Embedding(vocab_size,embed_size)
        self.lstm = nn.LSTM(input_size = embed_size , hidden_size = hidden_size , num_layers = num_layers , batch_first = True)
        self.linear = nn.Linear(hidden_size , vocab_size)
        
        self.softmax = nn.LogSoftmax(dim=1)
        
    
    def forward(self, features, captions):
        
        #remove tokens
        captions = captions[: , :-1]
        
        #get word embeddings
        caption_embeds = self.caption_embeddings(captions)
        
        # concatenate the feature and caption embeds
        inputs = torch.cat((features.unsqueeze(1),caption_embeds),1)
        
        #get output from lstm
        output , (h,c) = self.lstm(inputs)
        
        #get the predictions
        output = self.linear(output)
        
        return output

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        tokens = []
        for i in range(max_len) :
            out , states = self.lstm(inputs,states)
            out = self.linear(out.squeeze(1))
            predicted = out.max(dim=1)[1]
            tokens.append(predicted.item())
            
            inputs = self.caption_embeddings(predicted)
            inputs = inputs.unsqueeze(1)
            
        return tokens
            
 