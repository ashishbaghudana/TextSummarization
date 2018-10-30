from __future__ import unicode_literals, division, print_function
from argparse import ArgumentParser
import io
import os
import errno
import json

FULLTEXT_EXT = ".text"
SUMMARY_EXT = ".summary"

parser = ArgumentParser("Process News Summary dataset")
parser.add_argument(
    "-d",
    "--dataset",
    help="Path to dataset file",
    default="./news_summary.csv")
parser.add_argument(
    "-o", "--output", help="Path to output directory", default=".")

args = parser.parse_args()

try:
    fulltext_dir = os.path.join(args.output, 'fulltext')
    summary_dir = os.path.join(args.output, 'summary')
    os.makedirs(fulltext_dir)
    os.makedirs(summary_dir)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

with io.open(args.dataset, encoding='utf-8', errors='ignore') as freader:
    for index, row in enumerate(freader):
        content = json.loads(row)
        with open(os.path.join(fulltext_dir, f"{index}{FULLTEXT_EXT}"),
                  'w') as fp:
            fp.write(content['textbody'])
        with open(os.path.join(summary_dir, f"{index}{SUMMARY_EXT}"),
                  'w') as fp:
            fp.write(content['introduction'])
