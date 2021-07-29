
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
 
'''NPY数据格式'''
class MyDataset(Dataset):
    def __init__(self, data):
        self.data = np.load(data) #加载npy数据

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
def main():
    dataset=MyDataset('b.npy')
    data= DataLoader(dataset, batch_size=10, shuffle=True, num_workers=1)
    for maps in data:
        print(maps)

if __name__ == '__main__':
	main()