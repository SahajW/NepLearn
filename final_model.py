import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')


class QuestionPaperGenerator:
    """
    Final improved version with:
    - Automatic min_score_threshold (0.35)
    - Max 10 papers enforced
    - Smart fallback for depleted sources
    """

    def __init__(self, json_file_path, target_year):
        self.json_file_path = json_file_path
        self.target_year = target_year
        self.MAX_PAPERS = 10  # Hard limit
        self.MIN_SCORE = 0.35  # Automatic quality threshold
        self.max_topic_per_section = 2
        self.max_topic_total = 3
        self.models = {}
        self.df_labeled = None
        self.feature_cols = None
        self.global_used_questions = set()
        self.generated_papers = []
        self.exam_depleted = False
        self.textbook_depleted = False

    def load_and_flatten_json(self):
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
        num_cols = df.select_dtypes(include=[np.number]).columns
        df[num_cols] = df[num_cols].fillna(0)
        df["section"] = df["section"].fillna("UNKNOWN")

        exam_df = df[(df["source"] == "exam") & (df["year"] > 0)]
        concept_last_seen = exam_df.groupby("concept_id")["year"].max().to_dict()
        df["concept_last_seen_year"] = df["concept_id"].map(concept_last_seen).fillna(0)

        print(f"✓ Loaded {len(df)} questions")
        print(f"  Exam: {len(df[df['source'] == 'exam'])} | Textbook: {len(df[df['source'] != 'exam'])}")
        return df

    def create_labels(self, df, use_soft_labels=True, threshold=0.5):
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

            min_p = df.loc[tb_mask, "appear_probability"].min()
            max_p = df.loc[tb_mask, "appear_probability"].max()
            if max_p > min_p:
                df.loc[tb_mask, "appear_probability"] = (
                    (df.loc[tb_mask, "appear_probability"] - min_p) / (max_p - min_p)
                )

            df.loc[tb_mask, "label"] = (
                df.loc[tb_mask, "appear_probability"] > threshold
            ).astype(int)

        print(f"✓ Created {df['label'].sum()} positive labels")
        return df

    def engineer_features(self, df):
        data = df.copy()
        baseline_year = self.target_year - 4

        data["concept_last_seen_year"] = data["concept_last_seen_year"].replace(0, baseline_year)
        data["concept_years_since_last_seen"] = (
            self.target_year - data["concept_last_seen_year"]
        ).clip(lower=0, upper=10)

        data["concept_recency_decay"] = np.exp(-0.2 * data["concept_years_since_last_seen"])
        data["importance_x_concept_recency"] = (
            data["concept_importance"] * data["concept_recency_decay"]
        )

        data["is_exam_source"] = (data["source"] == "exam").astype(int)
        data["high_topic_importance"] = (data["topic_importance"] > data["topic_importance"].median()).astype(int)
        data["high_section_conf"] = (data["section_confidence"] > data["section_confidence"].median()).astype(int)
        data["concept_x_section"] = data["concept_importance"] * data["section_confidence"]

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

        return X, y, feature_cols, data

    def train_models(self, X_train, y_train):
        rf = RandomForestClassifier(
            n_estimators=100, max_depth=4, min_samples_leaf=8,
            min_samples_split=15, class_weight="balanced",
            random_state=42, n_jobs=-1,
        )

        xgb = XGBClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            subsample=0.7, colsample_bytree=0.7, min_child_weight=5,
            scale_pos_weight=min(10, (y_train == 0).sum() / max((y_train == 1).sum(), 1)),
            eval_metric="logloss", random_state=42,
        )

        rf.fit(X_train, y_train)
        xgb.fit(X_train, y_train)
        return {"RandomForest": rf, "XGBoost": xgb}

    def predict_probabilities(self, X):
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

        return np.clip(ensemble_score, 0.0, 1.0)

    def fit(self):
        print(f"\n{'='*80}")
        print(f"TRAINING MODEL FOR YEAR {self.target_year}")
        print(f"{'='*80}")
        print(f"Automatic settings:")
        print(f"  - Min score threshold: {self.MIN_SCORE}")
        print(f"  - Max papers allowed: {self.MAX_PAPERS}")
        print(f"{'='*80}\n")
        
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

        X_train, y_train = X[train_mask.values], y[train_mask.values]
        self.models = self.train_models(X_train, y_train)

        self.df_labeled["prediction_score"] = self.predict_probabilities(X)
        
        # Apply quality threshold
        before_filter = len(self.df_labeled)
        self.df_labeled = self.df_labeled[
            self.df_labeled["prediction_score"] >= self.MIN_SCORE
        ].copy()
        after_filter = len(self.df_labeled)
        
        print(f"\n✓ Models trained successfully")
        print(f"✓ Filtered to high-quality questions: {after_filter}/{before_filter} (removed {before_filter - after_filter})")
        print(f"{'='*80}\n")
        return self

    def remove_similar_questions(self, questions_df, threshold=0.75, used_ids=None):
        if len(questions_df) == 0:
            return questions_df

        used_ids = used_ids or set()
        if used_ids:
            questions_df = questions_df[~questions_df["id"].isin(used_ids)].copy()

        if len(questions_df) == 0:
            return questions_df

        questions_df = questions_df.drop_duplicates(subset=["raw_text"], keep="first")

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
                if sim[i][j] > threshold:
                    if valid.iloc[i]["prediction_score"] >= valid.iloc[j]["prediction_score"]:
                        to_remove.add(j)
                    else:
                        to_remove.add(i)
                        break

        keep_idx = [i for i in range(len(valid)) if i not in to_remove]
        return valid.iloc[keep_idx]

    def generate_question_paper(self, structure, similarity_threshold=0.75,
                                paper_number=1, exam_textbook_ratio=0.3):
        if self.df_labeled is None:
            raise ValueError("Call fit() first")

        all_used = self.global_used_questions.copy()
        filtered_df = self.df_labeled.copy()
        filtered_df = filtered_df.sort_values("prediction_score", ascending=False)
        filtered_df["text_length"] = filtered_df["raw_text"].str.len()

        paper = {
            "metadata": {
                "target_year": self.target_year,
                "paper_number": paper_number,
                "exam_textbook_ratio": exam_textbook_ratio,
                "warnings": []
            },
            "sections": {}
        }

        local_used_questions = set()
        local_used_topics = []

        for section_name, cfg in structure.items():
            available = filtered_df[~filtered_df["id"].isin(all_used)].copy()

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
                section_questions, threshold=similarity_threshold, used_ids=all_used
            )

            if len(section_questions) == 0:
                warning = f"❌ No questions available for {section_name}"
                paper["metadata"]["warnings"].append(warning)
                paper["sections"][section_name] = {
                    "instruction": cfg.get("instruction", ""),
                    "questions": []
                }
                continue

            # Calculate targets
            target_count = cfg["count"]
            target_exam = int(np.round(target_count * exam_textbook_ratio))
            target_textbook = target_count - target_exam

            exam_q = section_questions[section_questions["source"] == "exam"].copy()
            textbook_q = section_questions[section_questions["source"] != "exam"].copy()

            actual_exam_available = len(exam_q)
            actual_textbook_available = len(textbook_q)

            # Smart fallback logic
            if actual_exam_available == 0 and actual_textbook_available > 0:
                if not self.exam_depleted:
                    warning = f"⚠️ EXAM questions depleted! Using textbook questions only."
                    paper["metadata"]["warnings"].append(warning)
                    self.exam_depleted = True
                target_exam = 0
                target_textbook = target_count
            elif actual_textbook_available == 0 and actual_exam_available > 0:
                if not self.textbook_depleted:
                    warning = f"⚠️ TEXTBOOK questions depleted! Using exam questions only."
                    paper["metadata"]["warnings"].append(warning)
                    self.textbook_depleted = True
                target_exam = target_count
                target_textbook = 0
            elif actual_exam_available == 0 and actual_textbook_available == 0:
                warning = f"❌ CRITICAL: All questions depleted for {section_name}!"
                paper["metadata"]["warnings"].append(warning)
                paper["sections"][section_name] = {
                    "instruction": cfg.get("instruction", ""),
                    "questions": []
                }
                continue
            else:
                # Partial depletion
                if actual_exam_available < target_exam:
                    shortage = target_exam - actual_exam_available
                    target_exam = actual_exam_available
                    target_textbook = min(target_count - target_exam, actual_textbook_available)

                if actual_textbook_available < target_textbook:
                    shortage = target_textbook - actual_textbook_available
                    target_textbook = actual_textbook_available
                    target_exam = min(target_count - target_textbook, actual_exam_available)

            selected_dicts = []
            topic_count = {}

            # Select exam questions
            exam_count = 0
            for _, q in exam_q.iterrows():
                if exam_count >= target_exam:
                    break
                tid = q.get("topic_id", "unknown")
                if (topic_count.get(tid, 0) < self.max_topic_per_section and 
                    local_used_topics.count(tid) < self.max_topic_total):
                    selected_dicts.append(q.to_dict())
                    topic_count[tid] = topic_count.get(tid, 0) + 1
                    exam_count += 1

            # Select textbook questions
            textbook_count = 0
            for _, q in textbook_q.iterrows():
                if textbook_count >= target_textbook:
                    break
                tid = q.get("topic_id", "unknown")
                if (topic_count.get(tid, 0) < self.max_topic_per_section and 
                    local_used_topics.count(tid) < self.max_topic_total):
                    selected_dicts.append(q.to_dict())
                    topic_count[tid] = topic_count.get(tid, 0) + 1
                    textbook_count += 1

            # Fill remaining with ANY available questions
            while len(selected_dicts) < target_count:
                selected_ids = {q['id'] for q in selected_dicts}
                remaining_textbook = textbook_q[~textbook_q["id"].isin(selected_ids)]
                remaining_exam = exam_q[~exam_q["id"].isin(selected_ids)]

                found = False
                # Try both sources
                for source_df in [remaining_textbook, remaining_exam]:
                    if len(source_df) == 0:
                        continue
                    for _, q in source_df.iterrows():
                        tid = q.get("topic_id", "unknown")
                        if (topic_count.get(tid, 0) < self.max_topic_per_section and 
                            local_used_topics.count(tid) < self.max_topic_total):
                            selected_dicts.append(q.to_dict())
                            topic_count[tid] = topic_count.get(tid, 0) + 1
                            found = True
                            break
                    if found:
                        break

                if not found:
                    # Relax topic constraints
                    for source_df in [remaining_textbook, remaining_exam]:
                        if len(source_df) > 0:
                            q = source_df.iloc[0]
                            selected_dicts.append(q.to_dict())
                            tid = q.get("topic_id", "unknown")
                            topic_count[tid] = topic_count.get(tid, 0) + 1
                            found = True
                            break

                if not found:
                    warning = f"{section_name}: Only got {len(selected_dicts)}/{target_count} questions"
                    paper["metadata"]["warnings"].append(warning)
                    break

            selected_df = pd.DataFrame(selected_dicts).head(cfg["count"])

            if len(selected_df) > 0:
                local_used_questions.update(selected_df["id"].tolist())
                local_used_topics.extend(selected_df["topic_id"].tolist())
                self.global_used_questions.update(selected_df["id"].tolist())

            questions_list = []
            for idx, (_, row) in enumerate(selected_df.iterrows(), 1):
                questions_list.append({
                    "number": idx,
                    "text": row["raw_text"],
                    "source": row["source"],
                    "prediction_score": float(row["prediction_score"]),
                })

            paper["sections"][section_name] = {
                "instruction": cfg.get("instruction", ""),
                "questions": questions_list
            }

        return paper

    def generate_multiple_papers(self, num_papers, structure, exam_textbook_ratio=0.3):
        """Generate multiple papers with validation"""
        
        # Validate number of papers
        if num_papers > self.MAX_PAPERS:
            print(f"\n❌ ERROR: Cannot generate more than {self.MAX_PAPERS} papers!")
            print(f"   Requested: {num_papers}")
            print(f"   Maximum allowed: {self.MAX_PAPERS}")
            print(f"\n💡 Tip: Question sets more than {self.MAX_PAPERS} are not available.\n")
            return []
        
        if num_papers < 1:
            print("\n❌ ERROR: Need at least 1 paper!\n")
            return []

        print(f"\n{'='*80}")
        print(f"GENERATING {num_papers} PAPER{'S' if num_papers > 1 else ''} FOR {self.target_year}")
        print(f"{'='*80}")
        print(f"Settings:")
        print(f"  - Target ratio: {exam_textbook_ratio*100:.0f}% exam, {(1-exam_textbook_ratio)*100:.0f}% textbook")
        print(f"  - Similarity threshold: 0.75")
        print(f"  - Min quality score: {self.MIN_SCORE}")
        print(f"{'='*80}\n")

        papers = []
        for i in range(num_papers):
            print(f"[{i+1}/{num_papers}] Generating Paper #{i+1}...")
            paper = self.generate_question_paper(
                structure, 0.75, i+1, exam_textbook_ratio
            )
            papers.append(paper)
            self.generated_papers.append(paper)

            # Show stats
            exam_qs = sum(1 for s in paper["sections"].values() 
                         for q in s["questions"] if q["source"] == "exam")
            textbook_qs = sum(1 for s in paper["sections"].values() 
                             for q in s["questions"] if q["source"] != "exam")
            total_qs = exam_qs + textbook_qs

            if total_qs > 0:
                print(f"  ✓ {total_qs} questions: {exam_qs} exam ({exam_qs/total_qs*100:.1f}%), "
                      f"{textbook_qs} textbook ({textbook_qs/total_qs*100:.1f}%)")

                if paper["metadata"]["warnings"]:
                    print(f"  ⚠️  {len(paper['metadata']['warnings'])} warning(s)")
            else:
                print(f"  ❌ No questions generated!")

        print(f"\n{'='*80}")
        print(f"✅ Successfully generated {len(papers)} paper{'s' if len(papers) > 1 else ''}")
        print(f"{'='*80}\n")

        return papers

    def display_question_paper(self, paper):
        meta = paper.get("metadata", {})

        print("=" * 80)
        print(f"PAPER #{meta.get('paper_number', '?')} - YEAR {meta.get('target_year', '?')}")
        print(f"Target ratio: {meta.get('exam_textbook_ratio', 0)*100:.0f}% exam")
        print("=" * 80)

        if meta.get('warnings'):
            print("\n⚠️  WARNINGS:")
            for w in meta['warnings']:
                print(f"  - {w}")
            print()

        for section_name, section_data in paper.get("sections", {}).items():
            print(f"\n{section_name}")
            if section_data.get("instruction"):
                print(section_data["instruction"])
            print("-" * 80)

            questions = section_data.get("questions", [])
            if not questions:
                print("  (No questions available)")
            else:
                for q in questions:
                    print(f"\n{q['number']}. {q['text']}")
                    print(f"   [{q['source'].upper()}] Score: {q['prediction_score']:.4f}")

        print("\n" + "=" * 80)

    def display_all_papers(self, papers):
        for i, paper in enumerate(papers, 1):
            self.display_question_paper(paper)
            if i < len(papers):
                print("\n" * 2)

    def save_papers_to_json(self, papers, filename="generated_papers.json"):
        output = {
            "metadata": {
                "target_year": self.target_year,
                "total_papers": len(papers),
                "generation_date": str(pd.Timestamp.now()),
                "min_score_threshold": self.MIN_SCORE,
            },
            "papers": papers
        }
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"\n✅ Saved {len(papers)} paper(s) to {filename}")

    def get_stats(self):
        return {
            "papers_generated": len(self.generated_papers),
            "questions_used": len(self.global_used_questions),
            "questions_remaining": len(self.df_labeled) - len(self.global_used_questions),
            "target_year": self.target_year,
            "exam_depleted": self.exam_depleted,
            "textbook_depleted": self.textbook_depleted,
        }

# User input
print("\n" + "="*80)
print("EXAM QUESTION PAPER GENERATOR")
print("="*80 + "\n")

target_year = int(input("Enter target year (e.g., 2025): "))

# Initialize and train
predictor = QuestionPaperGenerator("final_data.json", target_year)
predictor.fit()


# Paper structure (based on COMP.pdf)
paper_structure = {
    "SECTION B": {
        "count": 7,
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

# Generate papers
num_papers = int(input("How many papers to generate (max 10): "))

exam_ratio_input = input("Exam ratio (0.0-1.0, press Enter for default 0.3): ").strip()
exam_ratio = float(exam_ratio_input) if exam_ratio_input else 0.3

papers = predictor.generate_multiple_papers(
    num_papers, 
    paper_structure, 
    exam_textbook_ratio=exam_ratio
)

# Display all generated papers
if papers:
    predictor.display_all_papers(papers)

# Show statistics
if papers:
    stats = predictor.get_stats()
    print(f"\n{'='*80}")
    print("📊 FINAL STATISTICS")
    print(f"{'='*80}")
    print(f"Papers generated: {stats['papers_generated']}")
    print(f"Questions used: {stats['questions_used']}")
    print(f"Questions remaining: {stats['questions_remaining']}")
    print(f"\nSource status:")
    print(f"  Exam questions depleted: {'YES ❌' if stats['exam_depleted'] else 'NO ✅'}")
    print(f"  Textbook questions depleted: {'YES ❌' if stats['textbook_depleted'] else 'NO ✅'}")
    print(f"{'='*80}\n")


