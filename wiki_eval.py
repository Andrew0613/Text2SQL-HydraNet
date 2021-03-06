#!/usr/bin/env python
import json
from argparse import ArgumentParser
from tqdm import tqdm
from wikisql_lib.dbengine import DBEngine
from wikisql_lib.query import Query
from wikisql_lib.common import count_lines


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--source_file', default='data/test.jsonl',help='source file for the prediction')
    parser.add_argument('--db_file', default='data/test.db',help='source database for the prediction')
    parser.add_argument('--pred_file', default='results/exp/test_out_eg.jsonl',help='predictions by the model')
    parser.add_argument('--ordered', action='store_true', help='whether the exact match should consider the order of conditions')
    args = parser.parse_args()

    engine = DBEngine(args.db_file)
    exact_match = []
    with open(args.source_file) as fs, open(args.pred_file) as fp:
        grades = []
        for ls, lp in tqdm(zip(fs, fp), total=count_lines(args.source_file)):
            eg = json.loads(ls)
            ep = json.loads(lp)
            qg = Query.from_dict(eg['sql'], ordered=args.ordered)
            gold = engine.execute_query(eg['table_id'], qg, lower=True)
            pred = ep.get('error', None)
            qp = None
            if not ep.get('error', None):
                try:
                    qp = Query.from_dict(ep['query'], ordered=args.ordered)
                    pred = engine.execute_query(eg['table_id'], qp, lower=True)
                except Exception as e:
                    pred = repr(e)
            correct = pred == gold
            match = qp == qg
            grades.append(correct)
            exact_match.append(match)
        print(json.dumps({
            'ex_accuracy': sum(grades) / len(grades),
            'lf_accuracy': sum(exact_match) / len(exact_match),
            }, indent=2))
