import re
from collections import Counter

import pandas as pd
from transformers import AutoTokenizer


class DataLoader:
    def __init__(self, file_path_folio, file_path_main, file_path_malls):
        self.file_path_folio = file_path_folio
        self.file_path_willow = file_path_main
        self.file_path_malls = file_path_malls

    def load_data(self):
        df_folio = pd.read_json(self.file_path_folio, lines=True)
        df_willow = pd.read_parquet(self.file_path_willow)
        df_malls = pd.read_json(self.file_path_malls)
        return df_folio, df_willow, df_malls

    @staticmethod
    def _split_premises(df_folio, nl_col="premises", fol_col="premises-FOL",
                         story_col="story_id", delimiter="\n"):
        deduped = df_folio.drop_duplicates(subset=[story_col, nl_col, fol_col])

        records = []
        mismatches = 0
        for _, row in deduped.iterrows():
            nl_lines = [s.strip() for s in str(row[nl_col]).split(delimiter) if s.strip()]
            fol_lines = [s.strip() for s in str(row[fol_col]).split(delimiter) if s.strip()]
            if len(nl_lines) != len(fol_lines):
                mismatches += 1
                continue
            for nl, fol in zip(nl_lines, fol_lines):
                records.append({"NL": nl, "FOL": fol, "Source": "FOLIO_Premise"})
        if mismatches:
            print(f"[FOLIO_Premise] skipped {mismatches} row(s): NL/FOL line-count mismatch")
        return pd.DataFrame(records)

    def get_data(self):
        df_folio, df_willow, df_malls = self.load_data()

        folio_premises = self._split_premises(df_folio)
        folio_conclusions = pd.DataFrame({
            'NL': df_folio['conclusion'],
            'FOL': df_folio['conclusion-FOL'],
            'Source': 'FOLIO_Conclusion'
        })

        willow_data = pd.DataFrame({
            'NL': df_willow['NL_sentence'],
            'FOL': df_willow['FOL_expression'],
            'Source': 'WILLOW'
        })
        malls_data = pd.DataFrame({
            'NL': df_malls['NL'],
            'FOL': df_malls['FOL'],
            'Source': 'MALLS'
        })

        df_combined = pd.concat(
            [folio_premises, folio_conclusions, willow_data, malls_data],
            ignore_index=True
        )
        df_combined = df_combined.dropna(subset=['NL', 'FOL'])
        df_combined['NL'] = df_combined['NL'].astype(str)
        df_combined['FOL'] = df_combined['FOL'].astype(str)
        df_combined = df_combined[
            (df_combined['NL'].str.strip() != '') & (df_combined['FOL'].str.strip() != '')
        ]
        return df_combined.reset_index(drop=True)


def discover_symbols(fol_strings):
    pattern = re.compile(r"[^\w\s]")
    counter = Counter()
    for s in fol_strings:
        counter.update(pattern.findall(s))
    return counter


class FOLTokenizerPipeline:
    MODEL_NAME = "bert-base-uncased"   # same base vocab for both sides

    def __init__(self, df):
        self.df = df
        self.df['NL'] = self.df['NL'].str.lower()
        self.df['FOL'] = self.df['FOL'].str.lower()

        self.nl_tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        self.fol_tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)

    def add_discovered_symbols(self, min_frequency=1):
        symbol_counts = discover_symbols(self.df['FOL'].tolist())
        existing_vocab = self.fol_tokenizer.get_vocab()

        to_add = []
        for sym, freq in symbol_counts.items():
            if freq < min_frequency:
                continue
            ids = self.fol_tokenizer.encode(sym, add_special_tokens=False)
            if len(ids) != 1 or sym not in existing_vocab:
                to_add.append(sym)

        added = self.fol_tokenizer.add_tokens(to_add)
        print(f"Discovered {len(symbol_counts)} distinct symbols, added {added} new tokens.")
        print("Symbol frequency:", symbol_counts.most_common())
        return symbol_counts

    def tokenize(self):
        self.df['NL_ids'] = self.df['NL'].apply(
            lambda x: self.nl_tokenizer.encode(x)
        )
        self.df['FOL_ids'] = self.df['FOL'].apply(
            lambda x: self.fol_tokenizer.encode(x)
        )
        return self.df

    def run_all(self):
        self.add_discovered_symbols()
        self.tokenize()
        return self.df

    def save(self, nl_dir="nl_tokenizer", fol_dir="fol_tokenizer"):
        self.nl_tokenizer.save_pretrained(nl_dir)
        self.fol_tokenizer.save_pretrained(fol_dir)


if __name__ == "__main__":
    loader = DataLoader(
        file_path_folio="<file_path>",
        file_path_main="<file_path>",
        file_path_malls="<file_path>"
    )
    df = loader.get_data()

    pipeline = FOLTokenizerPipeline(df)
    df = pipeline.run_all()
    pipeline.save()

    print("NL vocab size :", len(pipeline.nl_tokenizer), "(unchanged base bert-base-uncased vocab)")
    print("FOL vocab size:", len(pipeline.fol_tokenizer), "(base vocab + discovered symbols)")
    print(df[['NL', 'FOL', 'NL_ids', 'FOL_ids']].head())  # pyright: ignore[reportAttributeAccessIssue]

    nl_lens = df['NL'].apply(lambda x: len(pipeline.nl_tokenizer.encode(x))) # pyright: ignore[reportAttributeAccessIssue]
    fol_lens = df['FOL'].apply(lambda x: len(pipeline.fol_tokenizer.encode(x))) # pyright: ignore[reportAttributeAccessIssue]
    print("total NL tokens :", nl_lens.sum())
    print("total FOL tokens:", fol_lens.sum())

    def token_stats(df, nl_ids_col="NL_ids", fol_ids_col="FOL_ids"):
        nl_lens = df[nl_ids_col].apply(len)
        fol_lens = df[fol_ids_col].apply(len)
        return {
            "n_rows": len(df),
            "total_nl_tokens": int(nl_lens.sum()),
            "total_fol_tokens": int(fol_lens.sum()),
            "total_tokens": int(nl_lens.sum() + fol_lens.sum()),
            "nl_avg_len": float(nl_lens.mean()),
            "nl_max_len": int(nl_lens.max()),
            "fol_avg_len": float(fol_lens.mean()),
            "fol_max_len": int(fol_lens.max()),
        }

    print(token_stats(df))
