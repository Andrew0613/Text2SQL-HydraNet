import torch
import numpy as np
import utils
from torch import nn
import os
import time
import sys
from options import parse_args
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
from dataset.dataset import SQLDataset
class HydraModel(nn.Module):
    def __init__(self, config,device):
        super(HydraModel, self).__init__()
        self.config = config
        self.model = HydraNet(config)
        self.device = device
        self.model.to(self.device)
        if torch.cuda.device_count() > 1 and device == "cuda":
            self.model = nn.DataParallel(self.model,device_ids=config.gpu_ids)
    def save(self, model_path, epoch):
        save_path = os.path.join(model_path, "model_{0}.pt".format(epoch))
        if torch.cuda.device_count() > 1:
            torch.save(self.model.state_dict(), save_path)
        else:
            torch.save(self.model.state_dict(), save_path)
        print("Model saved in path: %s" % save_path)

    def load(self, model_path, epoch):
        pt_path = os.path.join(model_path, "model_{0}.pt".format(epoch))
        loaded_dict = torch.load(pt_path, map_location=torch.device(self.device))
        if torch.cuda.device_count() > 1:
            self.model.load_state_dict(loaded_dict)
        else:
            self.model.load_state_dict(loaded_dict)
        print("Model loaded from {0}".format(pt_path))
    def dataset_inference(self, dataset: SQLDataset):
        print("model prediction start")
        start_time = time.time()
        model_outputs = self.model_inference(dataset.model_inputs)

        final_outputs = []
        for pos in dataset.pos:
            final_output = {}
            for k in model_outputs:
                final_output[k] = model_outputs[k][pos[0]:pos[1], :]
            final_outputs.append(final_output)
        print("model prediction end, time elapse: {0}".format(time.time() - start_time))
        assert len(dataset.input_features) == len(final_outputs)

        return final_outputs
    def model_inference(self, model_inputs):
        self.model.eval()
        model_outputs = {}
        batch_size = 512
        for start_idx in range(0, model_inputs["input_ids"].shape[0], batch_size):
            input_tensor = {k: torch.from_numpy(model_inputs[k][start_idx:start_idx+batch_size]).to(self.device) for k in ["input_ids", "input_mask", "segment_ids"]}
            with torch.no_grad():
                model_output = self.model(**input_tensor)
            for k, out_tensor in model_output.items():
                if out_tensor is None:
                    continue
                if k not in model_outputs:
                    model_outputs[k] = []
                model_outputs[k].append(out_tensor.cpu().detach().numpy())

        for k in model_outputs:
            model_outputs[k] = np.concatenate(model_outputs[k], 0)

        return model_outputs
    def predict_SQL(self, dataset: SQLDataset, model_outputs=None):
        if model_outputs is None:
            model_outputs = self.dataset_inference(dataset)
        sqls = []
        for input_feature, model_output in zip(dataset.input_features, model_outputs):
            agg, select, where, conditions = self.parse_output(input_feature, model_output, [])

            conditions_with_value_texts = []
            for wc in where:
                _, op, vs, ve = conditions[wc]
                word_start, word_end = input_feature.subword_to_word[wc][vs], input_feature.subword_to_word[wc][ve]
                char_start = input_feature.word_to_char_start[word_start]
                char_end = len(input_feature.question)
                if word_end + 1 < len(input_feature.word_to_char_start):
                    char_end = input_feature.word_to_char_start[word_end + 1]
                value_span_text = input_feature.question[char_start:char_end].rstrip()
                conditions_with_value_texts.append((wc, op, value_span_text))

            sqls.append((agg, select, conditions_with_value_texts))

        return sqls

    def predict_SQL_with_EG(self, engine, dataset: SQLDataset, beam_size=5, model_outputs=None):
        if model_outputs is None:
            model_outputs = self.dataset_inference(dataset)
        sqls = []
        for input_feature, model_output in zip(dataset.input_features, model_outputs):
            agg, select, where_num, conditions = self.beam_parse_output(input_feature, model_output, beam_size)
            query = {"agg": agg, "sel": select, "conds": []}
            wcs = set()
            conditions_with_value_texts = []
            for condition in conditions:
                if len(wcs) >= where_num:
                    break
                _, wc, op, vs, ve = condition
                if wc in wcs:
                    continue

                word_start, word_end = input_feature.subword_to_word[wc][vs], input_feature.subword_to_word[wc][ve]
                char_start = input_feature.word_to_char_start[word_start]
                char_end = len(input_feature.question)
                if word_end + 1 < len(input_feature.word_to_char_start):
                    char_end = input_feature.word_to_char_start[word_end + 1]
                value_span_text = input_feature.question[char_start:char_end].rstrip()

                query["conds"] = [[int(wc), int(op), value_span_text]]
                result, sql = engine.execute_dict_query(input_feature.table_id, query)
                if not result or 'ERROR: ' in result:
                    continue

                conditions_with_value_texts.append((wc, op, value_span_text))
                wcs.add(wc)

            sqls.append((agg, select, conditions_with_value_texts))

        return sqls

    def _get_where_num(self, output):
        relevant_prob = 1 - np.exp(output["column_func"][:, 2])
        where_num_scores = np.average(output["where_num"], axis=0, weights=relevant_prob)
        where_num = int(np.argmax(where_num_scores))

        return where_num

    def parse_output(self, input_feature, model_output, where_label = []):
        def get_span(i):
            offset = 0
            segment_ids = np.array(input_feature.segment_ids[i])
            for j in range(len(segment_ids)):
                if segment_ids[j] == 1:
                    offset = j
                    break

            value_start, value_end = model_output["value_start"][i, segment_ids == 1], model_output["value_end"][i, segment_ids == 1]
            l = len(value_start)
            sum_mat = value_start.reshape((l, 1)) + value_end.reshape((1, l))
            span = (0, 0)
            for cur_span, _ in sorted(np.ndenumerate(sum_mat), key=lambda x:x[1], reverse=True):
                if cur_span[1] < cur_span[0] or cur_span[0] == l - 1 or cur_span[1] == l - 1:
                    continue
                span = cur_span
                break

            return (span[0]+offset, span[1]+offset)

        select_id_prob = sorted(enumerate(model_output["column_func"][:, 0]), key=lambda x:x[1], reverse=True)
        select = select_id_prob[0][0]
        agg = np.argmax(model_output["agg"][select, :])

        where_id_prob = sorted(enumerate(model_output["column_func"][:, 1]), key=lambda x:x[1], reverse=True)
        where_num = self._get_where_num(model_output)
        where = [i for i, _ in where_id_prob[:where_num]]
        conditions = {}
        for idx in set(where + where_label):
            span = get_span(idx)
            op = np.argmax(model_output["op"][idx, :])
            conditions[idx] = (idx, op, span[0], span[1])

        return agg, select, where, conditions

    def beam_parse_output(self, input_feature, model_output, beam_size=5):
        def get_span(i):
            offset = 0
            segment_ids = np.array(input_feature.segment_ids[i])
            for j in range(len(segment_ids)):
                if segment_ids[j] == 1:
                    offset = j
                    break

            value_start, value_end = model_output["value_start"][i, segment_ids == 1], model_output["value_end"][i, segment_ids == 1]
            l = len(value_start)
            sum_mat = value_start.reshape((l, 1)) + value_end.reshape((1, l))
            spans = []
            for cur_span, sum_logp in sorted(np.ndenumerate(sum_mat), key=lambda x:x[1], reverse=True):
                if cur_span[1] < cur_span[0] or cur_span[0] == l - 1 or cur_span[1] == l - 1:
                    continue
                spans.append((cur_span[0]+offset, cur_span[1]+offset, sum_logp))
                if len(spans) >= beam_size:
                    break

            return spans

        select_id_prob = sorted(enumerate(model_output["column_func"][:, 0]), key=lambda x:x[1], reverse=True)
        select = select_id_prob[0][0]
        agg = np.argmax(model_output["agg"][select, :])

        where_id_prob = sorted(enumerate(model_output["column_func"][:, 1]), key=lambda x:x[1], reverse=True)
        where_num = self._get_where_num(model_output)
        conditions = []
        for idx, wlogp in where_id_prob[:beam_size]:
            op = np.argmax(model_output["op"][idx, :])
            for span in get_span(idx):
                conditions.append((wlogp+span[2], idx, op, span[0], span[1]))
        conditions.sort(key=lambda x:x[0], reverse=True)
        return agg, select, where_num, conditions
class HydraNet(nn.Module):
    def __init__(self, config):
        super(HydraNet, self).__init__()
        self.config = config
        self.base_model = utils.create_base_model(config)

        # #=====Hack for RoBERTa model====
        # self.base_model.config.type_vocab_size = 2
        # single_emb = self.base_model.embeddings.token_type_embeddings
        # self.base_model.embeddings.token_type_embeddings = torch.nn.Embedding(2, single_emb.embedding_dim)
        # self.base_model.embeddings.token_type_embeddings.weight = torch.nn.Parameter(single_emb.weight.repeat([2, 1]), requires_grad=True)
        # #====================================

        drop_rate = float(config.dropout_rate)
        self.dropout = nn.Dropout(drop_rate)

        bert_hid_size = self.base_model.config.hidden_size
        self.column_func = nn.Linear(bert_hid_size, 3)
        self.agg = nn.Linear(bert_hid_size, int(config.agg_num))
        self.op = nn.Linear(bert_hid_size, int(config.op_num))
        self.where_num = nn.Linear(bert_hid_size, int(config.where_column_num) + 1)
        self.start_end = nn.Linear(bert_hid_size, 2)
    def forward(self, input_ids, input_mask, segment_ids, agg=None, select=None, where=None, where_num=None, op=None, value_start=None, value_end=None):
        # print("[inner] input_ids size:", input_ids.size())
        if self.config.model == "roberta":
            bert_output, pooled_output = self.base_model(
                input_ids=input_ids,
                attention_mask=input_mask,
                token_type_ids=None,
                return_dict=False)
        else:
            bert_output, pooled_output = self.base_model(
                input_ids=input_ids,
                attention_mask=input_mask,
                token_type_ids=segment_ids,
                return_dict=False)

        bert_output = self.dropout(bert_output)
        pooled_output = self.dropout(pooled_output)

        column_func_logit = self.column_func(pooled_output)
        agg_logit = self.agg(pooled_output)
        op_logit = self.op(pooled_output)
        where_num_logit = self.where_num(pooled_output)
        start_end_logit = self.start_end(bert_output)
        value_span_mask = input_mask.to(dtype=bert_output.dtype)
        # value_span_mask[:, 0] = 1
        start_logit = start_end_logit[:, :, 0] * value_span_mask - 1000000.0 * (1 - value_span_mask)
        end_logit = start_end_logit[:, :, 1] * value_span_mask - 1000000.0 * (1 - value_span_mask)

        loss = None
        if select is not None:
            bceloss = nn.BCEWithLogitsLoss(reduction="none")
            cross_entropy = nn.CrossEntropyLoss(reduction="none")

            loss = cross_entropy(agg_logit, agg) * select.float()
            loss += bceloss(column_func_logit[:, 0], select.float())
            loss += bceloss(column_func_logit[:, 1], where.float())
            loss += bceloss(column_func_logit[:, 2], (1-select.float()) * (1-where.float()))
            loss += cross_entropy(where_num_logit, where_num)
            loss += cross_entropy(op_logit, op) * where.float()
            loss += cross_entropy(start_logit, value_start)
            loss += cross_entropy(end_logit, value_end)


        # return loss, column_func_logit, agg_logit, op_logit, where_num_logit, start_logit, end_logit
        log_sigmoid = nn.LogSigmoid()

        return {"column_func": log_sigmoid(column_func_logit),
                "agg": agg_logit.log_softmax(1),
                "op": op_logit.log_softmax(1),
                "where_num": where_num_logit.log_softmax(1),
                "value_start": start_logit.log_softmax(1),
                "value_end": end_logit.log_softmax(1),
                "loss": loss}
class HydraEvaluator():
    def __init__(self, output_path, config, model:HydraModel, note=""):
        self.config = config
        self.model = model
        self.eval_history_file = os.path.join(output_path, "eval.log")
        self.bad_case_dir = os.path.join(output_path, "bad_cases")
        if not os.path.exists(self.bad_case_dir):
            os.mkdir(self.bad_case_dir)
        with open(self.eval_history_file, "w", encoding="utf8") as f:
            f.write(note.rstrip() + "\n")

        self.eval_data = {}
        val_path = os.path.join(config.dataroot,config.val_path)
        test_path = os.path.join(config.dataroot,config.test_path)
        for eval_path in [val_path, test_path]:
            eval_data = SQLDataset(eval_path, config, True)
            self.eval_data[os.path.basename(eval_path)] = eval_data

            print("Eval Data file {0} loaded, sample num = {1}".format(eval_path, len(eval_data)))

    def _eval_imp(self, eval_data: SQLDataset, get_sq=True):
        items = ["overall", "agg", "sel", "wn", "wc", "op", "val"]
        acc = {k:0.0 for k in items}
        sq = []
        cnt = 0
        model_outputs = self.model.dataset_inference(eval_data)
        for input_feature, model_output in zip(eval_data.input_features, model_outputs):
            cur_acc = {k:1 for k in acc if k != "overall"}

            select_label = np.argmax(input_feature.select)
            agg_label = input_feature.agg[select_label]
            wn_label = input_feature.where_num[0]
            wc_label = [i for i, w in enumerate(input_feature.where) if w == 1]

            agg, select, where, conditions = self.model.parse_output(input_feature, model_output, wc_label)
            if agg != agg_label:
                cur_acc["agg"] = 0
            if select != select_label:
                cur_acc["sel"] = 0
            if len(where) != wn_label:
                cur_acc["wn"] = 0
            if set(where) != set(wc_label):
                cur_acc["wc"] = 0

            for w in wc_label:
                _, op, vs, ve = conditions[w]
                if op != input_feature.op[w]:
                    cur_acc["op"] = 0

                if vs != input_feature.value_start[w] or ve != input_feature.value_end[w]:
                    cur_acc["val"] = 0

            for k in cur_acc:
                acc[k] += cur_acc[k]

            all_correct = 0 if 0 in cur_acc.values() else 1
            acc["overall"] += all_correct

            if ("DEBUG" in self.config or get_sq) and not all_correct:
                try:
                    true_sq = input_feature.output_SQ()
                    pred_sq = input_feature.output_SQ(agg=agg, sel=select, conditions=[conditions[w] for w in where])
                    task_cor_text = "".join([str(cur_acc[k]) for k in items if k in cur_acc])
                    sq.append([str(cnt), input_feature.question, "|".join([task_cor_text, pred_sq, true_sq])])
                except:
                    pass
            cnt += 1

        result_str = []
        for item in items:
            result_str.append(item + ":{0:.1f}".format(acc[item] * 100.0 / cnt))

        result_str = ", ".join(result_str)

        return result_str, sq

    def eval(self, epochs):
        print(self.bad_case_dir)
        for eval_file in self.eval_data:
            result_str, sq = self._eval_imp(self.eval_data[eval_file])
            print(eval_file + ": " + result_str)

            if "DEBUG" in self.config:
                for text in sq:
                    print(text[0] + ":" + text[1] + "\t" + text[2])
            else:
                with open(self.eval_history_file, "a+", encoding="utf8") as f:
                    f.write("[{0}, epoch {1}] ".format(eval_file, epochs) + result_str + "\n")

                bad_case_file = os.path.join(self.bad_case_dir,
                                           "{0}_epoch_{1}.log".format(eval_file, epochs))
                with open(bad_case_file, "w", encoding="utf8") as f:
                    for text in sq:
                        f.write(text[0] + ":" + text[1] + "\t" + text[2] + "\n")
if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    config = parse_args()
    # config["DEBUG"] = 1
    config.num_train_steps = 1000
    config.num_warmup_steps = 100
    device = torch.device("cuda:%s"%str(config.gpu_ids[0]) if torch.cuda.is_available() else "cpu")
    config = parse_args()
    str_ids = config.gpu_ids.split(',')
    config.gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            config.gpu_ids.append(id)
    device = torch.device("cuda:{}".format(config.gpu_ids[0]) if torch.cuda.is_available() else "cpu")
    model = HydraModel(config,device)
    evaluator = HydraEvaluator(config.checkpoints_dir, config, model, "debug evaluator")
    evaluator.eval(0)