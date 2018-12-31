import os
import argparse
import torch
import torch.nn as nn
from torchvision.transforms import transforms
import numpy as np
from datetime import datetime

from ImageLoader import ImageLoader
from model.cnn import ConvNet, ConvNet_half, ConvNet_ds
from model.ann import Ann, Ann_small
from logger import Logger

torch.manual_seed(2)

# argparse parameters
parser = argparse.ArgumentParser()
parser.add_argument('--model', dest='model', action='store',
                    default=None, help='Select model: Conv - ConvNet, ConvNet_half, ConvNet_ds ANN - Ann, Ann_small')
parser.add_argument('--save', dest='save_model', action='store',
                    default=None, help='Load already saved model')
parser.add_argument('--node', type=int, dest='node', action='store',
                    default=128, help='Number of nodes in each layers of ANN')
parser.add_argument('--mode', dest='mode', action='store',
                    default="train", help='If train or eval model')
parser.add_argument('--batchsize', type=int, dest='batchsize', action='store',
                    default=64, help='batchsize')
parser.add_argument('--maxepoch', type=int, dest='maxepoch', action='store',
                    default=200, help='Number of max epoches')
args = parser.parse_args()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

now = datetime.now()
NOWTIME = now.strftime("%Y%m%d-%H%M%S")

# Load data path
with open("data_path.txt", 'r') as file:
    file_list = list(file)
    TRAIN_DATASET_PATH = file_list[0].strip()
    TEST_DATASET_PATH = file_list[1].strip()

MODEL_SAVE_PATH = './save/' + args.model
MODEL_SAVE_RATE = 1
MODEL_LOG_PATH = './logs/' + args.model
T_B_GRAPH_PATH = './graph/' + args.model + NOWTIME

if args.save_model is not None:
    MODEL_SAVE_PATH = './save/' + args.save_model
    MODEL_SAVE_RATE = 1
    MODEL_LOG_PATH = './logs/' + args.save_model
    T_B_GRAPH_PATH = './graph/' + args.save_model + NOWTIME

BATCH_SIZE = args.batchsize
LEARNING_RATE = 0.0001

NUM_EPOCHS = args.maxepoch

is_write_sub_log = False

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))    # Normalize [-1, 1]
])

# train data set: image 1 x 57 x 116
train_dataset = ImageLoader(TRAIN_DATASET_PATH,
                            transforms=transform)

# data loader: batch_size x 1 x 57 x 116
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=BATCH_SIZE,
                                           shuffle=True)
# data set: image 1 x 57 x 116
test_dataset = ImageLoader(TEST_DATASET_PATH,
                            transforms=transform)

# data loader: batch_size x 1 x 57 x 116
test_loader = torch.utils.data.DataLoader(test_dataset,
                                          batch_size=BATCH_SIZE,
                                          shuffle=False)
# Network
if args.model == "Ann":
    net = Ann(node=args.node, num_classes=2).to(device)  # Real or Fake
elif args.model == "Ann_small":
    net = Ann_small(node=args.node, num_classes=2).to(device)
elif args.model == "ConvNet":
    net = ConvNet(num_classes=2).to(device)
elif args.model == "ConvNet_half":
    net = ConvNet_half(num_classes=2).to(device)
elif args.model == "ConvNet_ds":
    net = ConvNet_ds(num_classes=2).to(device)
else:
    raise ValueError('Wrong model')


# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)


def train():
    # train
    logger = Logger(T_B_GRAPH_PATH, is_train=True)

    total_step = len(train_loader)
    for epoch in range(NUM_EPOCHS):
        correct = 0
        losses = []
        main_accs = []
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = net(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # calculate accuracy
            _, pred_y = torch.max(outputs.data, 1)
            correct += (pred_y == labels).sum().item()
            main_acc = (correct / len(images)) * 100

            losses.append(loss)
            main_accs.append(main_acc)

            if i % BATCH_SIZE == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.8f}'
                      .format(epoch + 1, NUM_EPOCHS, i + 1, total_step, loss.item()))

        # write log
        logger.scalar_summary('loss', torch.mean(torch.FloatTensor(losses)), epoch + 1)
        logger.scalar_summary('Main_accuracy', torch.mean(torch.FloatTensor(main_accs)), epoch + 1)

        if is_write_sub_log:
            real_acc = logger.accuracy(net, TRAIN_DATASET_PATH, epoch, "Real", device)
            clay_acc = logger.accuracy(net, TRAIN_DATASET_PATH, epoch, "Clay", device)
            gltn_acc = logger.accuracy(net, TRAIN_DATASET_PATH, epoch, "Gltn", device)

            logger.scalar_summary('Real_accuracy', real_acc, epoch + 1)
            logger.scalar_summary('Clay_accuracy', clay_acc, epoch + 1)
            logger.scalar_summary('Gltn_accuracy', gltn_acc, epoch + 1)

        if epoch % MODEL_SAVE_RATE == 0:
            if not os.path.exists(MODEL_SAVE_PATH):
                os.makedirs(MODEL_SAVE_PATH)
            model_save_path = MODEL_SAVE_PATH + "/{}_epoch_{}.pth".format(args.model, NUM_EPOCHS)
            torch.save(net.state_dict(), model_save_path)


def eval():
    # load
    logger = Logger(T_B_GRAPH_PATH, is_train=False)

    eval_model_path = MODEL_SAVE_PATH + '/{}_epoch_{}.pth'.format(args.model, NUM_EPOCHS)
    print(eval_model_path)
    if os.path.exists(eval_model_path):
        net.load_state_dict(torch.load(eval_model_path))
    net.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)

    # train data accuracy
    logger.accuracy(net, TRAIN_DATASET_PATH, NUM_EPOCHS, "Real", device, MODEL_LOG_PATH, BATCH_SIZE, do_logwrite=True)
    logger.accuracy(net, TRAIN_DATASET_PATH, NUM_EPOCHS, "Clay", device, MODEL_LOG_PATH, BATCH_SIZE, do_logwrite=True)
    logger.accuracy(net, TRAIN_DATASET_PATH, NUM_EPOCHS, "Gltn", device, MODEL_LOG_PATH, BATCH_SIZE, do_logwrite=True)
    logger.accuracy(net, TRAIN_DATASET_PATH, NUM_EPOCHS, "", device, MODEL_LOG_PATH, BATCH_SIZE, do_logwrite=True)

    # test data accuracy
    logger.accuracy(net, TEST_DATASET_PATH, NUM_EPOCHS, "Real", device, MODEL_LOG_PATH, BATCH_SIZE, do_logwrite=True)
    logger.accuracy(net, TEST_DATASET_PATH, NUM_EPOCHS, "Clay", device, MODEL_LOG_PATH, BATCH_SIZE, do_logwrite=True)
    logger.accuracy(net, TEST_DATASET_PATH, NUM_EPOCHS, "Gltn", device, MODEL_LOG_PATH, BATCH_SIZE, do_logwrite=True)
    logger.accuracy(net, TEST_DATASET_PATH, NUM_EPOCHS, "", device, MODEL_LOG_PATH,BATCH_SIZE, do_logwrite=True)

    # FAR and FRR
    logger.FAR(net, TEST_DATASET_PATH, device, MODEL_LOG_PATH, BATCH_SIZE, do_logwrite=True)
    logger.FRR(net, TEST_DATASET_PATH, device, MODEL_LOG_PATH, BATCH_SIZE, do_logwrite=True)

    # Test the model
    with torch.no_grad():
        correct = 0
        total = 0
        true_positive = 0
        predict_true = 0
        real_true = 0

        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            _, pred_y = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (pred_y == labels).sum().item()

            true_positive += (pred_y * labels).sum().item()
            predict_true += pred_y.sum().item()
            real_true += labels.sum().item()

        accuracy = 100 * correct / total
        precision = (true_positive / (1e-9 + predict_true))
        recall = (true_positive / (1e-9 + real_true))
        f1score = (2 * precision * recall) / (precision + recall)

        print('Test Accuracy of the model on the test images: {} %'.format(accuracy))
        print('prec: {:.4f} recall: {:.4f} f1: {:.4f}'.format(precision, recall, f1score))

        with open('./logs/dataset__log.txt', 'a') as f:
            f.write('prec: {:.4f} recall: {:.4f} f1: {:.4f}\n'.format(precision, recall, f1score))

if __name__ == '__main__':
    if args.mode == 'train':
        train()
    elif args.mode == 'eval':
        eval()
