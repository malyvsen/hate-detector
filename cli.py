import argparse
import pandas as pd
import hate


parser = argparse.ArgumentParser()
parser.add_argument('input')
parser.add_argument('output')
parser.add_argument('--server')
args = parser.parse_args()

to_classify = pd.read_csv(args.input, names=['text'])
classes = hate.classify(to_classify.text.values, server=args.server)
classes = [int(c) for c in classes]
classes = pd.DataFrame.from_dict({'class': classes})
classes.to_csv(args.output, header=False, index=False)