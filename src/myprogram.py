#!/usr/bin/env python
import os
import string
import random
import pickle
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from datasets import load_dataset
from collections import defaultdict, Counter

class MyModel:
    """
    This is a starter model to get you started. Feel free to modify this file.
    """

    def __init__(self):
        self.n = 5
        self.model = defaultdict(Counter)
        self.char_freq = Counter()

    @classmethod
    def load_training_data(cls):
        print("Loading C4 dataset")
        
        dataset = load_dataset('allenai/c4', 'en', split='train', streaming=True)
        max_samples = 1000

        training_data = []
        sample_count = 0
        for sample in dataset:
            if sample_count >= max_samples:
                break
            if 'text' in sample:
                training_data.append(sample['text'])
                sample_count += 1
        
        print(f"Loaded {sample_count} samples")
        return training_data

    @classmethod
    def load_test_data(cls, fname):
        # your code here
        data = []
        with open(fname) as f:
            for line in f:
                inp = line[:-1]  # the last character is a newline
                data.append(inp)
        return data

    @classmethod
    def write_pred(cls, preds, fname):
        with open(fname, 'wt') as f:
            for p in preds:
                f.write('{}\n'.format(p))

    def run_train(self, data, work_dir):
        combined_text = ' '.join(data)
        n = self.n

        self.char_freq = Counter(combined_text)

        for i in range(len(combined_text) - n):
            for ctx_len in range(1, n + 1):
                if i + ctx_len < len(combined_text):
                    context = combined_text[i:i+ctx_len]
                    char = combined_text[i+ctx_len]
                    self.model[context][char] += 1
        
        print(f"Trained n-gram models")

    def predict_next_chars(self, inp):
        for ctx_len in range(min(self.n, len(inp)), 0, -1):
            context = inp[-ctx_len:]
            
            if context in self.model and self.model[context]:
                char_counts = self.model[context]
                top_3 = char_counts.most_common(3)
                pred = ''.join([char for char, count in top_3])

                # If less than 3 predictions, add most frequent chars
                if len(pred) < 3:
                    top_freq = self.char_freq.most_common(5)
                    for char, _ in top_freq:
                        if char not in pred and len(pred) < 3:
                            pred += char

                return pred

        top_freq = self.char_freq.most_common(3)
        return ''.join([char for char, _ in top_freq])

    def run_pred(self, data):
        preds = []
        for inp in data:
            pred = self.predict_next_chars(inp)
            preds.append(pred)
        return preds

    def save(self, work_dir):
        checkpoint = {
            'n': self.n,
            'model': dict(self.model),
            'char_freq': dict(self.char_freq)
        }
        with open(os.path.join(work_dir, 'model.checkpoint'), 'wb') as f:
            pickle.dump(checkpoint, f)

    @classmethod
    def load(cls, work_dir):
        with open(os.path.join(work_dir, 'model.checkpoint'), 'rb') as f:
            checkpoint = pickle.load(f)
        
        model = MyModel()
        model.n = checkpoint['n']
        model.model = defaultdict(Counter, {ctx: Counter(chars) for ctx, chars in checkpoint['model'].items()})
        model.char_freq = Counter(checkpoint.get('char_freq', {}))
        return model


if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('mode', choices=('train', 'test'), help='what to run')
    parser.add_argument('--work_dir', help='where to save', default='work')
    parser.add_argument('--test_data', help='path to test data', default='example/input.txt')
    parser.add_argument('--test_output', help='path to write test predictions', default='pred.txt')
    args = parser.parse_args()

    random.seed(0)

    if args.mode == 'train':
        if not os.path.isdir(args.work_dir):
            print('Making working directory {}'.format(args.work_dir))
            os.makedirs(args.work_dir)
        print('Instatiating model')
        model = MyModel()
        print('Loading training data')
        train_data = MyModel.load_training_data()
        print('Training')
        model.run_train(train_data, args.work_dir)
        print('Saving model')
        model.save(args.work_dir)
    elif args.mode == 'test':
        print('Loading model')
        model = MyModel.load(args.work_dir)
        print('Loading test data from {}'.format(args.test_data))
        test_data = MyModel.load_test_data(args.test_data)
        print('Making predictions')
        pred = model.run_pred(test_data)
        print('Writing predictions to {}'.format(args.test_output))
        assert len(pred) == len(test_data), 'Expected {} predictions but got {}'.format(len(test_data), len(pred))
        model.write_pred(pred, args.test_output)
    else:
        raise NotImplementedError('Unknown mode {}'.format(args.mode))
