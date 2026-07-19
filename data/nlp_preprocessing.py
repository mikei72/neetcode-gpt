import torch
import torch.nn as nn
from torchtyping import TensorType
from typing import List

class Solution:
    def get_dataset(self, positive: List[str], negative: List[str]) -> TensorType[float]:
        # 1. Build vocabulary: collect all unique words, sort them, assign integer IDs starting at 1
        # 2. Encode each sentence by replacing words with their IDs
        # 3. Combine positive + negative into one list of tensors
        # 4. Pad shorter sequences with 0s using nn.utils.rnn.pad_sequence(tensors, batch_first=True)
        combined = positive + negative

        vocab = []
        for s in combined:
            vocab.extend(s.split())
        vocab.sort()

        word_to_id = {}
        count = 1
        for word in vocab:
            if word not in word_to_id:
                word_to_id[word] = count
                count += 1
        
        encode = []
        for s in combined:
            enc = []
            for w in s.split():
                enc.append(word_to_id[w])
            encode.append(enc)
        encode = [torch.Tensor(s) for s in encode]
        
        return nn.utils.rnn.pad_sequence(encode, batch_first=True)

        

