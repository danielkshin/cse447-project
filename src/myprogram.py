#!/usr/bin/env python
import os
import string
import random
import pickle
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from datasets import load_dataset
from collections import defaultdict, Counter

class MyModel:
    def __init__(self):
        self.n = 5
        self.model = defaultdict(Counter)
        self.top3 = {}
        self.fallback3 = "   "

    @classmethod
    def load_training_data(cls):
        print("Loading C4 dataset")

        languages = {
            'en': 1500,
            'es': 850,
            'zh': 2000,
            'hi': 850,
            'pt': 850,
            'bn': 850,
            'ru': 850,
            'ja': 2000,
            'ar': 1500,
            'ko': 2000,
            'vi': 850,
            'fr': 850,
            'de': 850,
            'it': 1000,
            'tr': 850,
            'pl': 850,
            'nl': 850,
            'sv': 1000,
            'id': 850,
        }
        
        training_data = []
        
        for lang_code, max_samples in languages.items():
            print(f"Loading {lang_code} data")
            dataset = load_dataset('allenai/c4', lang_code, split='train', streaming=True)
            
            sample_count = 0
            for sample in dataset:
                if sample_count >= max_samples:
                    break
                if 'text' in sample:
                    training_data.append(sample['text'])
                    sample_count += 1
            print(f"Loaded {sample_count} samples.")

        return training_data

    @classmethod
    def load_test_data(cls, fname):
        # your code here
        data = []
        # need utf-8 to work on sophie's computer
        with open(fname, 'rt', encoding='utf-8') as f:
            for line in f:
                inp = line[:-1]  # the last character is a newline
                data.append(inp)
        return data

    @classmethod
    def write_pred(cls, preds, fname):
        # need utf-8 to work on sophie's computer
        with open(fname, 'wt', encoding='utf-8') as f:
            for p in preds:
                f.write('{}\n'.format(p))

    def run_train(self, data, work_dir):
        combined_text = ' '.join(data)
        n = self.n

        char_freq = Counter(combined_text)

        for i in range(len(combined_text) - n):
            for ctx_len in range(1, n + 1):
                if i + ctx_len < len(combined_text):
                    context = combined_text[i:i+ctx_len]
                    char = combined_text[i+ctx_len]
                    self.model[context][char] += 1

        self.build_fast_tables(char_freq)
        print(f"Trained n-gram models")

    def build_fast_tables(self, char_freq):
        exclude = {'\n', '\t'}

        fallback_chars = [ch for ch, _ in char_freq.most_common() if ch not in exclude]
        self.fallback3 = ''.join(fallback_chars[:3])

        top3 = {}
        for ctx, ctr in self.model.items():
            pred = [ch for ch, _ in ctr.most_common() if ch not in exclude][:3]
            pred_str = ''.join(pred)
            if len(pred_str) < 3:
                for ch in fallback_chars:
                    if ch not in pred_str:
                        pred_str += ch
                    if len(pred_str) == 3:
                        break
            top3[ctx] = pred_str

        self.top3 = top3

    def predict_next_chars(self, inp):
        top3 = self.top3
        n = self.n
        L = len(inp)
        max_len = n if L >= n else L

        for ctx_len in range(max_len, 0, -1):
            ctx = inp[L - ctx_len : L]
            pred = top3.get(ctx)
            if pred is not None:
                return pred

        return self.fallback3

    def run_pred(self, data):
        preds = []
        batch_size = 50
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size]
            for inp in batch:
                try:
                    pred = self.predict_next_chars(inp)
                    preds.append(pred)
                except Exception as e:
                    preds.append('   ')
        return preds

    def save(self, work_dir):
        checkpoint = {
            'n': self.n,
            'top3': self.top3,
            'fallback3': self.fallback3
        }
        with open(os.path.join(work_dir, 'model.checkpoint'), 'wb') as f:
            pickle.dump(checkpoint, f)

    @classmethod
    def load(cls, work_dir):
        with open(os.path.join(work_dir, 'model.checkpoint'), 'rb') as f:
            checkpoint = pickle.load(f)
        
        model = MyModel()
        model.n = checkpoint['n']
        model.top3 = checkpoint.get('top3', {})
        model.fallback3 = checkpoint.get('fallback3', '')
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
