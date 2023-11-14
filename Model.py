import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import numpy as np

from NetWork import ResNet
from ImageUtils import parse_record

""" This script defines the training, validation and testing process.
"""

class Cifar(nn.Module):
    def __init__(self, config):
        super(Cifar, self).__init__()
        self.config = config
        self.network = ResNet(
            self.config.resnet_version,
            self.config.resnet_size,
            self.config.num_classes,
            self.config.first_num_filters,
        )
        ### YOUR CODE HERE
        # define cross entropy loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(
            self.network.parameters(),
            lr=self.config.learning_rate,
            momentum=self.config.momentum,
            weight_decay=self.config.weight_decay
        )
        ### YOUR CODE HERE
    
    def make_batch(self, curr_x_train, batch_size):
        data = []
        for i in range(batch_size):
            image = parse_record(curr_x_train[i],training=True)
            data.append(image)
        return data


    def train(self, x_train, y_train, max_epoch):
        num_samples = x_train.shape[0]
        num_batches = int(num_samples/self.config.batch_size)
        print("Computed the num of batches ",num_batches)
        self.network.train()
        # Determine how many batches in an epoch
        print('### Training... ###')
        for epoch in range(1, max_epoch+1):
            start_time = time.time()
            # Shuffle
            shuffle_index = np.random.permutation(num_samples)
            curr_x_train = x_train[shuffle_index]
            curr_y_train = y_train[shuffle_index]
            ### YOUR CODE HERE
            # Set the learning rate for this epoch
            if epoch % 50 == 0:
                self.config.learning_rate = self.config.learning_rate/10            
            # Usage example: divide the initial learning rate by 10 after several epochs
            ### YOUR CODE HERE
            
            for i in range(num_batches):
                ### YOUR CODE HERE
                # Construct the current batch.
                # Don't forget to use "parse_record" to perform data preprocessing.
                # Don't forget L2 weight decay
                batch_start = i * self.config.batch_size
                batch_end = (i + 1) * self.config.batch_size
                inputs = self.make_batch(curr_x_train[batch_start:batch_end],self.config.batch_size)
                targets = curr_y_train[batch_start:batch_end]
                inputs = np.array(inputs)
                targets = np.array(targets)
                inputs = torch.tensor(inputs, dtype=torch.float32)
                targets = torch.tensor(targets, dtype=torch.long)
                #print("the shape of the batch before sending for training")
                #print(inputs.shape)
                #print(targets.shape)
                inputs = inputs.cuda()
                targets = targets.cuda()
                self.optimizer.zero_grad()
                outputs = self.network(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
                i+=1
                print('Batch {:d}/{:d} Loss {:.6f}'.format(i, num_batches, loss.item()), end='\r', flush=True)
            duration = time.time() - start_time
            print('Epoch {:d} Loss {:.6f} Duration {:.3f} seconds.'.format(epoch, loss.item(), duration))

            if epoch % self.config.save_interval == 0:
                self.save(epoch)


    def test_or_validate(self, x, y, checkpoint_num_list):
        self.network.eval()
        print('### Test or Validation ###')
        best_accuracy = 0.0
        best_checkpoint = None
        for checkpoint_num in checkpoint_num_list:
            checkpointfile = os.path.join(self.config.modeldir, 'model-%d.ckpt'%(checkpoint_num))
            self.load(checkpointfile)
            preds = []
            for i in tqdm(range(x.shape[0])):
                ### YOUR CODE HERE
                inputs = parse_record(x[i:i+1], training=False)
                inputs = torch.tensor(inputs, dtype=torch.float32)
                inputs = inputs.unsqueeze(0)
                inputs=inputs.cuda()
                outputs = self.network(inputs)
                _, predicted = torch.max(outputs, 1)
                preds.append(predicted.item())
                ### END CODE HERE
            y = torch.tensor(y)
            preds = torch.tensor(preds)
            accuracy = torch.sum(preds == y) / y.shape[0]
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_checkpoint = checkpoint_num
            print('Test accuracy: {:.4f}'.format(torch.sum(preds==y)/y.shape[0]))
        print('Best test accuracy: {:.4f} (from checkpoint {})'.format(best_accuracy, best_checkpoint))
        return best_checkpoint
    
    def save(self, epoch):
        checkpoint_path = os.path.join(self.config.modeldir, 'model-%d.ckpt'%(epoch))
        os.makedirs(self.config.modeldir, exist_ok=True)
        torch.save(self.network.state_dict(), checkpoint_path)
        print("Checkpoint has been created.")
    
    def load(self, checkpoint_name):
        ckpt = torch.load(checkpoint_name, map_location="cpu")
        self.network.load_state_dict(ckpt, strict=True)
        print("Restored model parameters from {}".format(checkpoint_name))