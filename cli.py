import argparse
import hate


parser = argparse.ArgumentParser()
parser.add_argument('input')
parser.add_argument('output')
parser.add_argument('--server')
args = parser.parse_args()

with open(args.input, 'r') as input:
    to_classify = input.read().split('\n')
if to_classify[-1] == '':
    to_classify = to_classify[:-1] # newline at end of file

classes = hate.classify(to_classify, server=args.server)
classes = [str(int(c)) for c in classes]
with open(args.output, 'w') as output:
    output.write('\n'.join(classes) + '\n')