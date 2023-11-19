import logging
from pathlib import Path
from typing import List, Dict, TextIO
import os
import json
import re

import fire
from tqdm import tqdm


class CombiningTranslator:

    def __init__(self, original_dir: str, output_dir: str, msmarco_dir: str, files: List[str]):
        self.original_dir = original_dir
        self.output_dir = output_dir
        self.msmarco_dir = msmarco_dir
        self.files = files
        self._queries = self._load_msmarco_queries()
        self._passages = self._load_msmarco_passages()

    def _load_msmarco_passages(self) -> Dict[str, str]:
        res = {}
        logging.info("Loading MS Marco corpus")
        collection_path = os.path.join(self.msmarco_dir, "collection.jsonl")
        pbar = tqdm(total=Path(collection_path).stat().st_size, unit='B', unit_scale=True, unit_divisor=1024)
        with open(collection_path, "r", encoding="utf-8") as input_file:
            for line in input_file:
                pbar.update(len(line.encode("utf-8")))
                value = json.loads(line.strip())
                qid = str(value["id"])
                res[qid] = value["translation"]
        pbar.close()
        return res

    def _load_msmarco_queries(self) -> Dict[str, str]:
        queries = {}
        logging.info("Loading MS Marco queries")
        for file_name in ("queries.dev.jsonl", "queries.eval.jsonl", "queries.train.jsonl"):
            file_path = os.path.join(self.msmarco_dir, file_name)
            with open(file_path, "r", encoding="utf-8") as input_file:
                for line in input_file:
                    value = json.loads(line.strip())
                    rowid = str(value["id"])
                    queries[rowid] = re.sub(r"[\n\r\t]+", " ", value["translation"]).lower()
        return queries

    def run(self, output_prefix: str):
        data_path = f"{output_prefix}_data.jsonl"
        perm_path = f"{output_prefix}_permutation.json"
        permutations: List[str] = []
        with open(data_path, "w", encoding="utf-8") as data_file:
            for file in self.files:
                logging.info(f"Converting file {file}")
                data_input = os.path.join(self.original_dir, file)
                perm_input = os.path.join(self.output_dir, file)
                self._run_for_input(data_input, perm_input, data_file, permutations)
        with open(perm_path, "w", encoding="utf-8") as perm_file:
            json.dump(permutations, perm_file)

    def _run_for_input(self, data_input: str, perm_input: str, data_output: TextIO, permutations: List[str]):
        with open(perm_input, "r", encoding="utf-8") as perm_file:
            perm_part = json.load(perm_file)
        data = []
        with open(data_input, "r", encoding="utf-8") as data_file:
            for line in data_file:
                value = json.loads(line.strip())
                qid = value["query_id"]
                query = self._queries[qid]
                positive_passages = self._convert_passages(value["positive_passages"])
                retrieved_passages = self._convert_passages(value["retrieved_passages"])
                output_value = {
                    "query": query,
                    "query_id": qid,
                    "positive_passages": positive_passages,
                    "retrieved_passages": retrieved_passages
                }
                data.append(output_value)
        assert len(data) == len(perm_part)
        for row in data:
            data_output.write(json.dumps(row, ensure_ascii=False))
            data_output.write("\n")
        for perm in perm_part:
            permutations.append(perm)

    def _convert_passages(self, passages: List[Dict]):
        output = []
        for passage in passages:
            docid = passage["docid"]
            text = self._passages[docid]
            output_value = {"docid": docid, "title": "-", "text": text}
            if "rank" in passage:
                output_value["rank"] = passage["rank"]
            output.append(output_value)
        return output


def run_app(original_dir: str, output_dir: str, msmarco_dir: str, files: str, output_prefix: str = "output"):
    file_list = [file.strip() for file in files.split(",")]
    translator = CombiningTranslator(original_dir, output_dir, msmarco_dir, file_list)
    translator.run(output_prefix)


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.INFO)
    logging.root.setLevel(logging.INFO)
    fire.Fire(run_app)
