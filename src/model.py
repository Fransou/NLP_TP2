import torch
from transformers import BertModel
import torch.nn as nn

from config import MAX_LEN, device

class CustomBERTModel(nn.Module):
    def __init__(self, bert_out_size = 12*3):
          super(CustomBERTModel, self).__init__()
          bert_model = BertModel.from_pretrained('bert-base-cased')
          self.bert = bert_model
          ### New layers:
          self.dropout0 = nn.Dropout(p = 0.5)
          self.linear1 = nn.Linear(self.bert.config.hidden_size*2, 1024)
          self.dropout1 = nn.Dropout(p = 0.4)
          self.linear2 = nn.Linear(1024, 1024)
          self.dropout2 = nn.Dropout(p = 0.3)
          self.linear3 = nn.Linear(1024, 1024)
          self.dropout3 = nn.Dropout(p = 0.3)
          self.linear4 = nn.Linear(1024, 1024)
          self.dropout4 = nn.Dropout(p = 0.3)
          self.linear5 = nn.Linear(1024, 1024)
          self.dropout5 = nn.Dropout(p = 0.3)
          self.linear6 = nn.Linear(1024, 1024)
          self.dropout6 = nn.Dropout(p = 0.3)
          self.linear7 = nn.Linear(1024, bert_out_size)
        
    def forward(self, input_ids, attention_mask, term_mask, aspect_id):
          out = self.bert(
              input_ids=input_ids,
              attention_mask=attention_mask
            )
          x = torch.mean(
              (out.last_hidden_state * term_mask.view(-1,MAX_LEN,1)),
              dim = 1
              ).to(torch.float32)
          x = torch.cat([x, out.pooler_output], dim=1)

          x = self.dropout0(x)
            
          x = nn.ReLU()(self.linear1(x))
          x = self.dropout1(x)
          x_skip = x
   
          x = self.linear2(x)  
          x = nn.ReLU()(x)
          x = self.dropout2(x)
            
          x = self.linear3(x)
          x = nn.ReLU()(x)
          x = self.dropout3(x)
            
          x = self.linear4(x + x_skip)
          x = nn.ReLU()(x)
          x = self.dropout4(x)
          x_skip = x
            
          x = self.linear5(x)
          x = nn.ReLU()(x)
          x = self.dropout5(x)
        
          x = self.linear6(x)
          x = nn.ReLU()(x)
          x = self.dropout6(x)
            
          x = self.linear7(x + x_skip)
        
          x =  x.view(-1,12,3)
          out = torch.Tensor([]).to(device)
          for i,id in enumerate(aspect_id):
              out = torch.cat([out, x[i,id,:].view(-1,3)])

          return nn.Softmax(dim=1)(out.view(-1,3))
    

    
