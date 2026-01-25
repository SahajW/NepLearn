import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity


class QuestionPaperPredictor:
    """
    Predicts likely exam questions using historical exam data and textbook questions.
    Trains RandomForest + XGBoost and generates a structured question paper.
    """

    def __init__(self, json_file_path, target_year):
        self.json_file_path = json_file_path
        self.target_year = target_year
        self.models = {}
        self.df_labeled = None
        self.feature_cols = None

    # ------------------------------------------------------------
    # DATA LOADING
    # ------------------------------------------------------------
    def load_and_flatten_json(self):
        """Load nested JSON and flatten into a DataFrame"""

        with open(self.json_file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        rows = []
        for q in data.get("questions", []):
            exam = q.get("exam_meta") or {}
            df_feat = q.get("derived_features") or {}

            section_probs = df_feat.get("section_prob", {})
            section_conf = max(section_probs.values()) if section_probs else 0

            row = {
                "id": q.get("id"),
                "subject": q.get("subject"),
                "source": q.get("source"),
                "raw_text": q.get("raw_text"),
                "embedding": q.get("embedding"),
                "topic_id": q.get("topic_id"),
                "concept_id": q.get("concept_id"),
                "section": exam.get("section"),
                "marks": exam.get("marks"),
                "year": int(exam["year"]) if exam.get("year") else 0,
                "year_semester": exam.get("year_semester"),
                "topic_importance": df_feat.get("topic_importance", 0),
                "concept_importance": df_feat.get("concept_importance", 0),
                "section_confidence": section_conf,
            }

            rows.append(row)

        df = pd.DataFrame(rows)

        # Fill missing values
        num_cols = df.select_dtypes(include=[np.number]).columns
        df[num_cols] = df[num_cols].fillna(0)
        df["section"] = df["section"].fillna("UNKNOWN")

        # Concept-level last seen year (from exam data only)
        exam_df = df[(df["source"] == "exam") & (df["year"] > 0)]
        concept_last_seen = exam_df.groupby("concept_id")["year"].max().to_dict()

        df["concept_last_seen_year"] = df["concept_id"].map(concept_last_seen).fillna(0)

        print(f"Loaded {len(df)} questions")
        print(f"Exam questions: {len(df[df['source'] == 'exam'])}")
        print(f"Textbook questions: {len(df[df['source'] != 'exam'])}")
        print(f"Concepts with exam history: {(df['concept_last_seen_year'] > 0).sum()}")

        return df

    # ------------------------------------------------------------
    # LABEL CREATION
    # ------------------------------------------------------------
    def create_labels(self, df, use_soft_labels=True, threshold=0.5):
        """
        Label = 1 for recent exam questions.
        Textbook questions get soft labels using importance + section confidence.
        """

        df = df.copy()
        df["label"] = 0

        exam_mask = (df["source"] == "exam") & (df["year"] >= self.target_year - 8)
        df.loc[exam_mask, "label"] = 1

        if use_soft_labels:
            tb_mask = df["source"] != "exam"

            df.loc[tb_mask, "appear_probability"] = (
                df.loc[tb_mask, "topic_importance"] * 0.45 +
                df.loc[tb_mask, "concept_importance"] * 0.45 +
                df.loc[tb_mask, "section_confidence"] * 0.10
            )

            # Min-max normalize to 0–1
            min_p = df.loc[tb_mask, "appear_probability"].min()
            max_p = df.loc[tb_mask, "appear_probability"].max()
            if max_p > min_p:
                df.loc[tb_mask, "appear_probability"] = (
                    (df.loc[tb_mask, "appear_probability"] - min_p) / (max_p - min_p)
                )

            df.loc[tb_mask, "label"] = (
                df.loc[tb_mask, "appear_probability"] > threshold
            ).astype(int)

        print(f"Positive labels: {df['label'].sum()}")
        return df

    # ------------------------------------------------------------
    # FEATURE ENGINEERING
    # ------------------------------------------------------------
    def engineer_features(self, df):
        """
        Creates concept-recency features and interaction terms.
        Textbook concepts are treated as if last seen 4 years ago.
        """

        data = df.copy()
        baseline_year = self.target_year - 4

        data["concept_last_seen_year"] = data["concept_last_seen_year"].replace(0, baseline_year)
        data["concept_years_since_last_seen"] = (
            self.target_year - data["concept_last_seen_year"]
        ).clip(lower=0, upper=10)

        # Exponential decay: recent concepts get higher weight
        data["concept_recency_decay"] = np.exp(-0.2 * data["concept_years_since_last_seen"])
        data["importance_x_concept_recency"] = (
            data["concept_importance"] * data["concept_recency_decay"]
        )

        data["is_exam_source"] = (data["source"] == "exam").astype(int)
        data["high_topic_importance"] = (data["topic_importance"] > data["topic_importance"].median()).astype(int)
        data["high_section_conf"] = (data["section_confidence"] > data["section_confidence"].median()).astype(int)
        data["concept_x_section"] = data["concept_importance"] * data["section_confidence"]

        # Section one-hot encoding (textbook treated as UNKNOWN)
        masked_section = data["section"].where(data["source"] == "exam", "UNKNOWN")
        section_dummies = pd.get_dummies(masked_section, prefix="section")
        data = pd.concat([data, section_dummies], axis=1)

        feature_cols = [
            "topic_importance",
            "concept_importance",
            "section_confidence",
            "is_exam_source",
            "concept_years_since_last_seen",
            "concept_recency_decay",
            "high_topic_importance",
            "high_section_conf",
            "concept_x_section",
            "importance_x_concept_recency",
        ] + section_dummies.columns.tolist()

        X = data[feature_cols]
        y = data["label"]

        print(f"Engineered {len(feature_cols)} features")
        return X, y, feature_cols, data

    # ------------------------------------------------------------
    # MODEL TRAINING
    # ------------------------------------------------------------
    def train_models(self, X_train, y_train):
        """Train RandomForest and XGBoost"""

        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=4,
            min_samples_leaf=8,
            min_samples_split=15,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )

        xgb = XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.7,
            colsample_bytree=0.7,
            min_child_weight=5,
            scale_pos_weight=min(10, (y_train == 0).sum() / max((y_train == 1).sum(), 1)),
            eval_metric="logloss",
            random_state=42,
        )

        rf.fit(X_train, y_train)
        xgb.fit(X_train, y_train)

        print("Models trained")
        return {"RandomForest": rf, "XGBoost": xgb}

    def evaluate_models(self, X_test, y_test):
        """Evaluate each model separately"""

        for name, model in self.models.items():
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            print(f"{name} -> Accuracy: {acc:.4f}, F1: {f1:.4f}")

    # ------------------------------------------------------------
    # PREDICTION
    # ------------------------------------------------------------
    def predict_probabilities(self, X):
        """Ensemble prediction with concept-recency boost for exam questions"""

        proba_dict = {name: model.predict_proba(X)[:, 1] for name, model in self.models.items()}
        ensemble_score = 0.5 * proba_dict["RandomForest"] + 0.5 * proba_dict["XGBoost"]

        if "concept_recency_decay" in X.columns and self.df_labeled is not None:
            source_mask = self.df_labeled["source"] == "exam"
            recency_boost = np.ones(len(X))

            exam_indices = np.where(source_mask)[0]
            if len(exam_indices) > 0:
                decay = X.loc[exam_indices, "concept_recency_decay"].values
                recency_boost[exam_indices] = 1 + (1 - decay) * 0.3

            ensemble_score *= recency_boost

        return ensemble_score

    # ------------------------------------------------------------
    # FULL PIPELINE
    # ------------------------------------------------------------
    def fit(self):
        """Load data, create labels, train models, generate predictions"""

        df_raw = self.load_and_flatten_json()
        self.df_labeled = self.create_labels(df_raw)
        X, y, self.feature_cols, _ = self.engineer_features(self.df_labeled)

        textbook_mask = self.df_labeled["source"] == "textbook"
        sampled_textbook = textbook_mask & (np.random.rand(len(self.df_labeled)) < 0.2)

        train_mask = (
            ((self.df_labeled["source"] == "exam") &
             (self.df_labeled["year"] < self.target_year - 2)) |
            sampled_textbook
        )

        test_mask = (
            (self.df_labeled["source"] == "exam") &
            (self.df_labeled["year"] >= self.target_year - 2) &
            (self.df_labeled["year"] < self.target_year)
        )

        X_train, y_train = X[train_mask.values], y[train_mask.values]
        X_test, y_test = X[test_mask.values], y[test_mask.values]

        self.models = self.train_models(X_train, y_train)

        if len(X_test) > 0:
            self.evaluate_models(X_test, y_test)

        self.df_labeled["prediction_score"] = self.predict_probabilities(X)
        self.X, self.y = X, y

        print("Training complete")
        return self

    # ------------------------------------------------------------
    # DUPLICATE FILTERING
    # ------------------------------------------------------------
    def remove_similar_questions(self, questions_df, similarity_threshold=0.85, used_texts=None):
        """Remove exact and embedding-based duplicate questions"""

        if len(questions_df) == 0:
            return questions_df

        used_texts = used_texts or set()
        questions_df = questions_df[~questions_df["raw_text"].isin(used_texts)].copy()
        questions_df = questions_df.drop_duplicates(subset=["raw_text"], keep="first")

        def text_signature(text):
            return " ".join(str(text).lower().split())[:150]

        questions_df["sig"] = questions_df["raw_text"].apply(text_signature)
        questions_df = questions_df.drop_duplicates(subset=["sig"], keep="first").drop(columns=["sig"])

        if "embedding" not in questions_df.columns:
            return questions_df

        valid = questions_df[questions_df["embedding"].notna()].copy()
        if len(valid) == 0:
            return questions_df

        embeddings = np.array(valid["embedding"].tolist())
        sim = cosine_similarity(embeddings)

        to_remove = set()
        for i in range(len(sim)):
            if i in to_remove:
                continue
            for j in range(i + 1, len(sim)):
                if sim[i][j] > similarity_threshold:
                    if valid.iloc[i]["prediction_score"] >= valid.iloc[j]["prediction_score"]:
                        to_remove.add(j)
                    else:
                        to_remove.add(i)
                        break

        keep_idx = [i for i in range(len(valid)) if i not in to_remove]
        return valid.iloc[keep_idx]

    # ------------------------------------------------------------
    # PAPER GENERATION
    # ------------------------------------------------------------
    def generate_question_paper(self, structure, similarity_threshold=0.85, source_filter="mixed"):
        """Generate a structured question paper from predictions (store only number & text)"""

        if self.df_labeled is None:
            raise ValueError("Call fit() first")

        if source_filter == "textbook":
            filtered_df = self.df_labeled[self.df_labeled["source"] == "textbook"].copy()
        elif source_filter == "exam":
            filtered_df = self.df_labeled[self.df_labeled["source"] == "exam"].copy()
        else:
            filtered_df = self.df_labeled.copy()

        filtered_df = filtered_df.sort_values("prediction_score", ascending=False)
        filtered_df["text_length"] = filtered_df["raw_text"].str.len()

        paper = {"metadata": {"target_year": self.target_year}, "sections": {}}
        used_questions, used_texts, used_topics = set(), set(), []

        for section_name, cfg in structure.items():
            available = filtered_df[~filtered_df["id"].isin(used_questions)].copy()

            if cfg.get("priority_topics"):
                pat = "|".join(cfg["priority_topics"])
                pr = available[available["raw_text"].str.lower().str.contains(pat, na=False)]
                if len(pr) > 0:
                    available = pr

            if cfg.get("require_multipart"):
                available = available[
                    (available["text_length"] > 150) &
                    (
                        (available["raw_text"].str.count(r"\.") > 2) |
                        (available["raw_text"].str.contains(
                            "write a program|differentiate|explain|describe",
                            case=False, na=False))
                    )
                ]

            min_len = cfg.get("min_length", 0)
            max_len = cfg.get("max_length", float("inf"))
            available = available[
                (available["text_length"] >= min_len) &
                (available["text_length"] <= max_len)
            ]

            sec_match = available[
                (available["section"] == section_name) |
                (available["section"] == "UNKNOWN")
            ]
            section_questions = sec_match if len(sec_match) > 0 else available

            section_questions = self.remove_similar_questions(
                section_questions, similarity_threshold, used_texts
            )

            selected = []
            topic_count = {}
            for _, q in section_questions.iterrows():
                tid = q.get("topic_id", "unknown")
                if topic_count.get(tid, 0) < 2 and used_topics.count(tid) < 3:
                    selected.append(q)
                    topic_count[tid] = topic_count.get(tid, 0) + 1
                if len(selected) >= cfg["count"]:
                    break

            selected = pd.DataFrame(selected).head(cfg["count"])

            if len(selected) > 0:
                used_questions.update(selected["id"].tolist())
                used_texts.update(selected["raw_text"].tolist())
                used_topics.extend(selected["topic_id"].tolist())

            # Only keep number and text
            questions_list = []
            for idx, (_, row) in enumerate(selected.iterrows(), 1):
                questions_list.append({
                    "number": idx,
                    "text": row["raw_text"],
                    "source": row["source"], 
                    "prediction_score": row["prediction_score"],
                })

            # Store with instruction
            paper["sections"][section_name] = {
                "instruction": structure.get(section_name, {}).get("instruction", ""),
                "questions": questions_list
            }

        return paper

    # ------------------------------------------------------------
    # DISPLAY PAPER
    # ------------------------------------------------------------
    def display_question_paper(self, question_paper):
        """
        Print the generated question paper in a readable format.
        Shows section name, question text, source, and prediction score.
        """

        meta = question_paper.get("metadata", {})
        sections = question_paper.get("sections", {})

        print("=" * 80)
        print(f"PREDICTED QUESTION PAPER - YEAR {meta.get('target_year', 'N/A')}")
        print("=" * 80)

        for section_name, section_data in sections.items():
            instruction = section_data.get("instruction", "")
            questions = section_data.get("questions", [])

            print(f"\n{section_name}")
            if instruction:
                print(instruction)
            print("-" * 80)

            for q in questions:
                number = q.get("number", "?")
                text = q.get("text", "")
                source = q.get("source", "N/A").upper()
                score = q.get("prediction_score", 0)
                compulsory = q.get("compulsory", False)

                flags = []
                if compulsory:
                    flags.append("COMPULSORY")

                flag_str = f" [{' | '.join(flags)}]" if flags else ""

                print(f"\n{number}. {text}{flag_str}")
                print(f"   Source: {source} | Score: {score:.4f}")

        print("\n" + "=" * 80)


#example usage
if __name__ == "__main__":

    predictor = QuestionPaperPredictor("final_data.json", target_year=2025)
    predictor.fit()

    paper_structure = {
        "SECTION B": {
            "count": 7,
            "compulsory": [4],
            "instruction": "Attempt Any SIX Questions",
            "min_length": 30,
            "max_length": 400,
        },
        "SECTION C": {
            "count": 3,
            "instruction": "Attempt Any TWO Questions",
            "min_length": 150,
            "priority_topics": ["structure", "struct", "file", "storage class", "dynamic memory"],
            "require_multipart": True,
        },
    }

    paper = predictor.generate_question_paper(paper_structure)
    predictor.display_question_paper(paper)

    # with open("generated_question_paper.json", "w", encoding="utf-8") as f:
    #     json.dump(paper, f, indent=2, ensure_ascii=False)

    # print("Question paper saved to generated_question_paper.json")
