import torch
from torch.utils.data import Dataset

class PlayDataset(Dataset):
    # transform is transformation function (if needed) to preprocess vectors, such as normalization or scaling
    def __init__(self, data_file, transform=None):
        super().__init__()
        self.data = self._load_data(data_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        vector = self.data[idx]['vector']
        label = self.data[idx]['label']

        if self.transform:
            vector = self.transform(vector)

        encoder_input = torch.tensor(vector, dtype=torch.float32)
        decoder_input = torch.tensor([label], dtype=torch.float32)

        return {
            "vector": vector, 
            "label": label,
            "encoder_input": encoder_input,  # (seq_len)
            "decoder_input": decoder_input,  # (seq_len)
            "encoder_mask": torch.ones_like(encoder_input.unsqueeze(0).unsqueeze(0), dtype=torch.int), # (1, 1, seq_len)
            "decoder_mask": decoder_input.unsqueeze(0).int() & torch.ones((1, 1, 1), dtype=torch.int), # (1, seq_len) & (1, seq_len, seq_len),
        }

    def _load_data(self, data_file):
        data = []
        with open(data_file, 'r') as file:
            for line in file:
                line = line.strip().split(',')
                vector = [float(x) for x in line[:-1]]  # Assuming the vector is comma-separated
                label = int(line[-1])  # Assuming the label is the last element
                data.append({'vector': vector, 'label': label})
        return data

    