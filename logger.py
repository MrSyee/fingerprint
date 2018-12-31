# summary function for drawing graph
# ref : https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/04-utils/tensorboard/logger.py

# Code referenced from https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514
import os
import tensorflow as tf
import numpy as np
import scipy.misc
import torch
from torchvision.transforms import transforms
from ImageLoader import ImageLoader

try:
    from StringIO import StringIO  # Python 2.7
except ImportError:
    from io import BytesIO  # Python 3.x


class Logger(object):
    def __init__(self, log_dir, is_train=False):
        """Create a summary writer logging to log_dir."""
        if is_train:
            self.writer = tf.summary.FileWriter(log_dir)

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        self.writer.add_summary(summary, step)

    def image_summary(self, tag, images, step):
        """Log a list of images."""

        img_summaries = []
        for i, img in enumerate(images):
            # Write the image to a string
            try:
                s = StringIO()
            except:
                s = BytesIO()
            scipy.misc.toimage(img).save(s, format="png")

            # Create an Image object
            img_sum = tf.Summary.Image(encoded_image_string=s.getvalue(),
                                       height=img.shape[0],
                                       width=img.shape[1])
            # Create a Summary value
            img_summaries.append(tf.Summary.Value(tag='%s/%d' % (tag, i), image=img_sum))

        # Create and write Summary
        summary = tf.Summary(value=img_summaries)
        self.writer.add_summary(summary, step)

    def histo_summary(self, tag, values, step, bins=1000):
        """Log a histogram of the tensor of values."""

        # Create a histogram using numpy
        counts, bin_edges = np.histogram(values, bins=bins)

        # Fill the fields of the histogram proto
        hist = tf.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values ** 2))

        # Drop the start of the first bin
        bin_edges = bin_edges[1:]

        # Add bin edges and counts
        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)

        # Create and write Summary
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
        self.writer.add_summary(summary, step)
        self.writer.flush()

    def accuracy(self, net, path, epoch, phase, device, log_path, batch_size=64, do_logwrite=False):
        try:
            os.makedirs(log_path)
        except OSError:
            pass

        category = path.split('/')[-1] + "_" + phase
        with torch.no_grad():
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))    # Normalize [-1, 1]
            ])

            dataset = ImageLoader(path, phase, transforms=transform)
            data_loader = torch.utils.data.DataLoader(dataset,
                                                      batch_size=batch_size,
                                                      shuffle=False)

            correct = 0
            for idx, (images, labels) in enumerate(data_loader):
                images = images.to(device)
                labels = labels.to(device)
                outputs = net(images)
                _, pred_y = torch.max(outputs.data, 1)

                correct += (pred_y == labels).sum().item()

            accuracy = (correct / len(dataset)) * 100

            print(
                'phase:%s --- correct [%d/%d]  acc: %.2f%%' %
                                                        (category, correct, len(dataset), accuracy))

            if do_logwrite:
                f = open('%s/%s_log.txt' % (log_path, category), 'a')
                f.write('epoch:%d, phase:%s --- correct [%d/%d]  %.4f%%\n' % (
                epoch, phase, correct, len(dataset), accuracy))
                f.close()

            return accuracy

    def FAR(self, net, path, device, log_path, batch_size=64, do_logwrite=False):
        try:
            os.makedirs(log_path)
        except OSError:
            pass

        with torch.no_grad():
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))    # Normalize [-1, 1]
            ])

            dataset = ImageLoader(path, "Fake", transforms=transform)
            data_loader = torch.utils.data.DataLoader(dataset,
                                                      batch_size=batch_size,
                                                      shuffle=False)

            wrong = 0
            for idx, (images, labels) in enumerate(data_loader):
                images = images.to(device)
                labels = labels.to(device)
                outputs = net(images)
                _, pred_y = torch.max(outputs.data, 1)
                wrong += (pred_y != labels).sum().item()
            metric = (wrong / len(dataset)) * 100
            print_log = '%s --- neg(fake) -> pos(real) Wrong [%d/%d]  FAR: %.2f%%' % ("FAR", wrong, len(dataset), metric)
            print(print_log)

        if do_logwrite:
            f = open('%s/%s_log.txt' % (log_path, "test_"), 'a')
            f.write(print_log)
            f.close()

        return metric


    def FRR(self, net, path, device, log_path, batch_size=64, do_logwrite=False):
        try:
            os.makedirs(log_path)
        except OSError:
            pass

        with torch.no_grad():
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))    # Normalize [-1, 1]
            ])

            dataset = ImageLoader(path, "Real", transforms=transform)
            data_loader = torch.utils.data.DataLoader(dataset,
                                                      batch_size=batch_size,
                                                      shuffle=False)

            wrong = 0
            for idx, (images, labels) in enumerate(data_loader):
                images = images.to(device)
                labels = labels.to(device)
                outputs = net(images)
                _, pred_y = torch.max(outputs.data, 1)
                wrong += (pred_y != labels).sum().item()
            metric = (wrong / len(dataset)) * 100
            print_log = '%s --- pos(real) -> neg(fake) Wrong [%d/%d]  FRR: %.2f%%' % ("FRR", wrong, len(dataset), metric)
            print(print_log)

        if do_logwrite:
            f = open('%s/%s_log.txt' % (log_path, "test_"), 'a')
            f.write(print_log)
            f.close()

        return metric
