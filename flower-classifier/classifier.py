# Imports here
import numpy as np
import time
import torch
import torchvision

from collections import OrderedDict
from torch import nn

class ImageClassifier():
    
    def __init__(self, device=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is not None else 'cpu'
        self._clf_list = ['classifier', 'fc']

        self.name = None
        self.model = None
        self.classifier_name = None
        
        self.criterion = None
        self.criterion_name = None
        self.optimizer = None
        self.optimizer_name = None
        self.n_classes = None
        self.n_hidden_units = None
        self.learningrate = None
        self.p_dropout = None
        
        self.class_to_idx = None
        self.idx_to_class = None
        self.class_to_labels = None
        
        self.epoch = 0
        self.training_duration = []
        self.training_loss = []
        self.validation_loss = []
        self.validation_accuracy = []

    def _clf_name(self):
        for attr in self._clf_list:
            clf = getattr(self.model, attr, None)
            name = attr
            if clf is not None:
                break
                
        if clf is None:
            raise AttributeError(f'{self.model} has none of these attributes {self._clf_list}.')
                
        return name

    def _n_inputs(self):
        n_inputs = None
        clf = getattr(self.model, self.classifier_name)
        
        try:
            n_inputs = clf[0].in_features
        except:
            try:
                n_inputs = clf.in_features
            except AttributeError as err:
                print(err)
            
        return n_inputs
    
    def _load_criterion(self, name):
        return getattr(torch.nn, name)()
    
    def _load_optimizer(self, name, parameters=None):
        if parameters is None:
            parameters = {}
        optimizer = getattr(torch.optim, name)

        return optimizer(getattr(self.model, self.classifier_name).parameters(), **parameters)

    def initialize(self, model_name, n_classes, criterion_name='NLLLoss', optimizer_name='Adam',
                   learningrate=0.001, n_hidden_units=2048, p_dropout=0.2):
        
        self.name = model_name
        self.model = getattr(torchvision.models, model_name)(pretrained=True)
        self.classifier_name = self._clf_name()

        for param in self.model.parameters():
            param.requires_grad = False
        
        if n_hidden_units < n_classes:
            raise ValueError(f'Minimum number of hidden units for model {self.name} is {n_classes}.')
            
        n_second_layer = int((n_hidden_units + n_classes) / 2)
        n_inputs = self._n_inputs()
        if n_hidden_units > n_inputs:
            raise ValueError(f'Maximum number of hidden units for model {self.name} is {n_inputs}.')
            
        clf = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(n_inputs, n_hidden_units)),
            ('relu1', nn.ReLU()),
            ('dropout1', nn.Dropout(p=p_dropout)),
            ('fc2', nn.Linear(n_hidden_units, n_second_layer)),
            ('relu2', nn.ReLU()),
            ('dropout2', nn.Dropout(p=p_dropout)),
            ('fc3', nn.Linear(n_second_layer, n_classes)),
            ('output', nn.LogSoftmax(dim=1))
        ]))
        
        setattr(self.model, self.classifier_name, clf)
        for param in getattr(self.model, self.classifier_name).parameters():
            param.requires_grad = True
        self.model = self.model.to(self.device)

        self.criterion = self._load_criterion(criterion_name)
        self.criterion_name = criterion_name
        self.optimizer = self._load_optimizer(optimizer_name, {'lr': learningrate})
        self.optimizer_name = optimizer_name
        self.n_classes = n_classes
        self.n_hidden_units = n_hidden_units
        self.learningrate = learningrate
        self.p_dropout = p_dropout

        return self.model

    def load_indexes_labels(self, labels, idx):
        self.class_to_labels = labels
        self.class_to_idx = idx
        self.idx_to_class = {val:key for key, val in idx.items()}
    
    def train(self, data):
        start = time.time()    
        running_loss = 0
        for images, labels in data:
            # Move data to the device used for processing the model
            images, labels = images.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()

            ps_log = self.model.forward(images)
            ps = torch.exp(ps_log)

            loss = self.criterion(ps_log, labels)
            loss.backward()

            self.optimizer.step()

            # Update metric
            running_loss += loss.item()
        
        self.epoch += 1
        
        duration = time.time() - start
        average_loss = running_loss/len(data)
        self.training_duration.append(duration)
        self.training_loss.append(average_loss)
        
        return average_loss

    def validate(self, data):
        # Set Model to Evaluation mode (faster)
        self.model.eval()
        with torch.no_grad():
            running_loss = 0
            running_accuracy = 0
            for images, labels in data:
                images, labels = images.to(self.device), labels.to(self.device)

                # Run model & get probabilities
                ps_log = self.model.forward(images)
                ps = torch.exp(ps_log)

                # Get most probable label per image & calculate accuracy of the predictions
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)

                # Update metrics
                loss = self.criterion(ps_log, labels)
                running_loss += loss.item()

                running_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

        # Reset Model to training mode
        self.model.train()

        average_loss = running_loss/len(data)
        accuracy = running_accuracy/len(data)
        self.validation_loss.append(average_loss)
        self.validation_accuracy.append(accuracy)

        return average_loss, accuracy
        
    def predict(self, image, topk=1, labels=None):
        ''' Predict the class (or classes) of an image using a trained deep learning model.
        '''
        self.model.eval()
        image = image.to(self.device).float()
        
        if labels is None:
            labels = self.class_to_labels
        
        with torch.no_grad():
            ps_log = self.model.forward(image)
            ps = torch.exp(ps_log)

            # Get 5 highest probabilities and classes
            probs, idx = ps.topk(topk, dim=1)
            probs = [round(p, 2) for p in probs[0].tolist()]

            class_idx = [self.idx_to_class[i] for i in idx[0].tolist()]
            class_labels = []
            for id in class_idx:
                class_labels.append(id if labels is None else labels[str(id)])

            order = np.argsort(probs)
            probs = [probs[x] for x in order]
            class_labels = [class_labels[x] for x in order]

        self.model.train()

        return reversed(list(zip(probs, class_labels)))
    
    def save(self, name):
        checkpoint = {'model_name': self.name,
                      'model_state_dict': self.model.state_dict(),
                      'classifier_name': self.classifier_name,
                      'optimizer_name': self.optimizer_name,
                      'optimizer_state_dict': self.optimizer.state_dict(),
                      'criterion_name': self.criterion_name,
                      'n_classes': self.n_classes,
                      'n_hidden_units': self.n_hidden_units,
                      'learningrate': self.learningrate,
                      'p_dropout': self.p_dropout,
                      'epoch': self.epoch,
                      'training_duration': self.training_duration,
                      'training_loss': self.training_loss,
                      'validation_loss': self.validation_loss,
                      'validation_accuracy': self.validation_accuracy,
                      'idx_to_class': self.idx_to_class,
                      'class_to_idx': self.class_to_idx,
                      'class_to_labels': self.class_to_labels}

        torch.save(checkpoint, name)
    
    def load(self, name):
        d = torch.load(name)

        self.initialize(d['model_name'], d['n_classes'], d['criterion_name'], d['optimizer_name'],
                        d['learningrate'], d['n_hidden_units'], d['p_dropout'])
        
        self.model.load_state_dict((d['model_state_dict']))
        self.optimizer.load_state_dict(d['optimizer_state_dict'])

        self.epoch = d['epoch']
        self.training_duration = d['training_duration']
        self.training_loss = d['training_loss']
        self.validation_loss = d['validation_loss']
        self.validation_accuracy = d['validation_accuracy']
        self.idx_to_class = d['idx_to_class']
        self.class_to_idx = d['class_to_idx']
        self.class_to_labels = d['class_to_labels']

        return self.model
