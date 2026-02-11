#!/usr/bin/env python
import os
import pickle
import random
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from collections import defaultdict, Counter
from datasets import load_dataset, interleave_datasets
import json


class MyModel:
    """
    Character-level n-gram model with backoff for next character prediction.
    Trained on C4 dataset (English, Korean, Vietnamese).
    """

    def __init__(self, n=5, max_training_chars=50_000_000):
        """
        Args:
            n: Maximum n-gram order (will use n-grams from 1 to n with backoff)
            max_training_chars: Maximum number of characters to process during training
        """
        self.n = n
        self.max_training_chars = max_training_chars
        self.ngram_counts = [defaultdict(Counter) for _ in range(n)]  # ngram_counts[i] stores (i+1)-grams
        self.char_set = set()
        
    @classmethod
    def load_training_data(cls):
        """
        Load training data from C4 dataset (English, Korean, Vietnamese).
        Uses streaming mode for memory efficiency.
        """
        print("Loading C4 dataset in streaming mode...")
        
        # Load the three languages in streaming mode
        # We'll manually interleave them to avoid schema mismatch issues
        datasets_list = []
        for lang in ["en", "ko", "vi"]:
            print(f"Loading {lang} dataset...")
            ds = load_dataset("allenai/c4", lang, split="train", streaming=True)
            datasets_list.append(ds)
        
        # Create a custom iterator that interleaves the datasets
        class InterleavedDataset:
            def __init__(self, datasets):
                self.iterators = [iter(ds) for ds in datasets]
                self.current_idx = 0
            
            def __iter__(self):
                return self
            
            def __next__(self):
                # Round-robin through the datasets
                max_tries = len(self.iterators)
                for _ in range(max_tries):
                    try:
                        item = next(self.iterators[self.current_idx])
                        self.current_idx = (self.current_idx + 1) % len(self.iterators)
                        return item
                    except StopIteration:
                        # If one dataset is exhausted, continue with the next
                        self.current_idx = (self.current_idx + 1) % len(self.iterators)
                raise StopIteration
        
        dataset = InterleavedDataset(datasets_list)
        return dataset

    @classmethod
    def load_test_data(cls, fname):
        """Load test data from file."""
        data = []
        with open(fname) as f:
            for line in f:
                inp = line.rstrip('\n')  # remove newline but keep other whitespace
                data.append(inp)
        return data

    @classmethod
    def write_pred(cls, preds, fname):
        """Write predictions to file."""
        with open(fname, 'wt') as f:
            for p in preds:
                f.write('{}\n'.format(p))

    def run_train(self, data, work_dir):
        """
        Train the n-gram model on streaming data.
        
        Args:
            data: Streaming dataset from Hugging Face
            work_dir: Directory to save training statistics
        """
        print(f"Training {self.n}-gram model...")
        
        chars_processed = 0
        docs_processed = 0
        
        for example in data:
            if chars_processed >= self.max_training_chars:
                break
                
            text = example['text']
            
            # Process the text
            for i in range(len(text)):
                char = text[i]
                self.char_set.add(char)
                
                # Build n-grams of different orders
                for order in range(self.n):
                    if i >= order:
                        # Get context of length 'order'
                        context = text[max(0, i-order):i]
                        # Record that 'char' follows 'context'
                        self.ngram_counts[order][context][char] += 1
            
            chars_processed += len(text)
            docs_processed += 1
            
            if docs_processed % 1000 == 0:
                print(f"Processed {docs_processed} documents, {chars_processed:,} characters")
        
        print(f"Training complete! Processed {docs_processed} documents, {chars_processed:,} characters")
        print(f"Vocabulary size: {len(self.char_set)} characters")
        
        # Save statistics
        stats = {
            'docs_processed': docs_processed,
            'chars_processed': chars_processed,
            'vocab_size': len(self.char_set),
            'n': self.n
        }
        
        with open(os.path.join(work_dir, 'train_stats.json'), 'w') as f:
            json.dump(stats, f, indent=2)

    def run_pred(self, data):
        """
        Make predictions using n-gram model with backoff.
        
        Args:
            data: List of input strings
            
        Returns:
            List of prediction strings (3 characters each)
        """
        preds = []
        
        for inp in data:
            top_guesses = self._predict_next_char(inp, top_k=3)
            preds.append(''.join(top_guesses))
        
        return preds
    
    def _predict_next_char(self, context, top_k=3):
        """
        Predict next character using n-gram model with backoff.
        
        Args:
            context: String context to predict from
            top_k: Number of predictions to return
            
        Returns:
            List of top k predicted characters
        """
        # Try n-grams from highest order to lowest (backoff strategy)
        for order in range(self.n - 1, -1, -1):
            # Get the relevant context for this order
            ctx = context[-order:] if order > 0 else ""
            
            if ctx in self.ngram_counts[order]:
                char_counts = self.ngram_counts[order][ctx]
                
                # Get top k most common characters
                most_common = char_counts.most_common(top_k)
                
                if most_common:
                    predictions = [char for char, count in most_common]
                    
                    # If we don't have enough predictions, fill with random from vocab
                    while len(predictions) < top_k:
                        # Try lower order n-grams first
                        if order > 0:
                            break  # Let backoff handle it
                        # If we're at unigrams and still need more, use random
                        random_char = random.choice(list(self.char_set)) if self.char_set else ' '
                        if random_char not in predictions:
                            predictions.append(random_char)
                    
                    if len(predictions) >= top_k:
                        return predictions[:top_k]
        
        # If all else fails, return most common characters from unigrams
        if self.ngram_counts[0][""]:
            most_common = self.ngram_counts[0][""].most_common(top_k)
            predictions = [char for char, count in most_common]
            if len(predictions) >= top_k:
                return predictions[:top_k]
        
        # Last resort: return random characters from vocabulary
        if self.char_set:
            return random.sample(list(self.char_set), min(top_k, len(self.char_set)))
        
        # Absolute fallback
        return [' ', 'e', 't'][:top_k]

    def save(self, work_dir):
        """Save model to disk."""
        print(f"Saving model to {work_dir}...")
        
        model_data = {
            'n': self.n,
            'max_training_chars': self.max_training_chars,
            'ngram_counts': self.ngram_counts,
            'char_set': self.char_set
        }
        
        with open(os.path.join(work_dir, 'model.checkpoint'), 'wb') as f:
            pickle.dump(model_data, f)
        
        print("Model saved successfully!")

    @classmethod
    def load(cls, work_dir):
        """Load model from disk."""
        print(f"Loading model from {work_dir}...")
        
        with open(os.path.join(work_dir, 'model.checkpoint'), 'rb') as f:
            model_data = pickle.load(f)
        
        model = cls(n=model_data['n'], max_training_chars=model_data['max_training_chars'])
        model.ngram_counts = model_data['ngram_counts']
        model.char_set = model_data['char_set']
        
        print("Model loaded successfully!")
        return model


if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('mode', choices=('train', 'test'), help='what to run')
    parser.add_argument('--work_dir', help='where to save', default='work')
    parser.add_argument('--test_data', help='path to test data', default='example/input.txt')
    parser.add_argument('--test_output', help='path to write test predictions', default='pred.txt')
    parser.add_argument('--n', type=int, default=5, help='n-gram order')
    parser.add_argument('--max_chars', type=int, default=50_000_000, 
                       help='maximum characters to process during training')
    args = parser.parse_args()

    random.seed(42)

    if args.mode == 'train':
        if not os.path.isdir(args.work_dir):
            print('Making working directory {}'.format(args.work_dir))
            os.makedirs(args.work_dir)
        print('Instantiating model')
        model = MyModel(n=args.n, max_training_chars=args.max_chars)
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
