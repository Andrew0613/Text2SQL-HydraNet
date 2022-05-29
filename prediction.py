import os
import json
import pickle
import utils
from wikisql_lib.dbengine import DBEngine
from dataset.dataset import SQLDataset
from model import HydraModel
import torch 
from options import parse_args
def print_metric(label_file, pred_file):
    sp = [(json.loads(ls)["sql"], json.loads(lp)["query"]) for ls, lp in zip(open(label_file), open(pred_file))]

    sel_acc = sum(p["sel"] == s["sel"] for s, p in sp) / len(sp)
    agg_acc = sum(p["agg"] == s["agg"] for s, p in sp) / len(sp)
    wcn_acc = sum(len(p["conds"]) == len(s["conds"]) for s, p in sp) / len(sp)

    def wcc_match(a, b):
        a = sorted(a, key=lambda k: k[0])
        b = sorted(b, key=lambda k: k[0])
        return [c[0] for c in a] == [c[0] for c in b]

    def wco_match(a, b):
        a = sorted(a, key=lambda k: k[0])
        b = sorted(b, key=lambda k: k[0])
        return [c[1] for c in a] == [c[1] for c in b]

    def wcv_match(a, b):
        a = sorted(a, key=lambda k: k[0])
        b = sorted(b, key=lambda k: k[0])
        return [str(c[2]).lower() for c in a] == [str(c[2]).lower() for c in b]

    wcc_acc = sum(wcc_match(p["conds"], s["conds"]) for s, p in sp) / len(sp)
    wco_acc = sum(wco_match(p["conds"], s["conds"]) for s, p in sp) / len(sp)
    wcv_acc = sum(wcv_match(p["conds"], s["conds"]) for s, p in sp) / len(sp)

    print('sel_acc: {}\nagg_acc: {}\nwcn_acc: {}\nwcc_acc: {}\nwco_acc: {}\nwcv_acc: {}\n' \
          .format(sel_acc, agg_acc, wcn_acc, wcc_acc, wco_acc, wcv_acc))


if __name__ == "__main__":
    config = parse_args()
    str_ids = config.gpu_ids.split(',')
    config.gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
           config.gpu_ids.append(id)
    if len(config.gpu_ids) > 0:
        torch.cuda.set_device(config.gpu_ids[0])
    in_file = os.path.join(config.dataroot, config.test_path)
    label_file = "data/test.jsonl"
    db_file = "data/test.db"
    result_path = os.path.join(config.results,config.name)
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    out_file = os.path.join(result_path, "test_out.jsonl")
    out_file_eg = os.path.join(result_path, "test_out_eg.jsonl")
    model_out_file = os.path.join(result_path, "test_model_out.pkl")
    # All Best
    model_path = os.path.join(config.checkpoints_dir,config.name)
    epoch = 4

    engine = DBEngine(db_file)
    pred_data = SQLDataset(in_file, config, False)
    print("num of samples: {0}".format(len(pred_data.input_features)))
    device = torch.device('cuda:{}'.format(config.gpu_ids[0])) if config.gpu_ids else torch.device('cpu')
    model =  HydraModel(config,device)
    model.load(model_path, epoch)
    if os.path.exists(model_out_file):
        model_outputs = pickle.load(open(model_out_file, "rb"))
    else:
        model_outputs = model.dataset_inference(pred_data)
        pickle.dump(model_outputs, open(model_out_file, "wb"))

    print("===HydraNet===")
    pred_sqls = model.predict_SQL(pred_data, model_outputs=model_outputs)
    with open(out_file, "w") as g:
        for pred_sql in pred_sqls:
            # print(pred_sql)
            result = {"query": {}}
            result["query"]["agg"] = int(pred_sql[0])
            result["query"]["sel"] = int(pred_sql[1])
            result["query"]["conds"] = [(int(cond[0]), int(cond[1]), str(cond[2])) for cond in pred_sql[2]]
            g.write(json.dumps(result) + "\n")
    print_metric(label_file, out_file)

    print("===HydraNet+EG===")
    pred_sqls = model.predict_SQL_with_EG(engine, pred_data, model_outputs=model_outputs)
    with open(out_file_eg, "w") as g:
        for pred_sql in pred_sqls:
            # print(pred_sql)
            result = {"query": {}}
            result["query"]["agg"] = int(pred_sql[0])
            result["query"]["sel"] = int(pred_sql[1])
            result["query"]["conds"] = [(int(cond[0]), int(cond[1]), str(cond[2])) for cond in pred_sql[2]]
            g.write(json.dumps(result) + "\n")
    print_metric(label_file, out_file + ".eg")
