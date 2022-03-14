import torch

MAX_LEN = 80

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")