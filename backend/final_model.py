"""
Question Paper Generator - Core Module
=======================================
A machine learning-based system for generating exam question papers.

Features:
- ML-based question prediction using Random Forest and XGBoost
- Smart question deduplication using cosine similarity
- Topic and concept tracking to ensure diversity
- Source balancing (exam vs textbook questions)
- Automatic quality filtering
"""

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
    ML-powered question paper generator with automatic quality control.
    
    Key Features:
    - Automatic min_score_threshold (0.35)
    - Max 10 papers enforced
    - Smart fallback for depleted sources
    - Topic diversity constraints
    - Similarity-based deduplication
    
    Parameters
    ----------
    json_file_path : str
        Path to the questions JSON file
    target_year : int
        Target year for question paper generation
    max_papers : int, optional
        Maximum number of papers allowed (default: 10)
    min_score : float, optional
        Minimum prediction score threshold (default: 0.35)
    max_topic_per_section : int, optional
        Maximum questions per topic in a section (default: 2)
    max_topic_total : int, optional
        Maximum questions per topic across all sections (default: 3)
    """

    def __init__(self, json_file_path, target_year, max_papers=10, min_score=0.35,
                 max_topic_per_section=2, max_topic_total=3):
        """Initialize the question paper generator."""
        self.json_file_path = json_file_path
        self.target_year = target_year
        self.MAX_PAPERS = max_papers
        self.MIN_SCORE = min_score
        self.max_topic_per_section = max_topic_per_section
        self.max_topic_total = max_topic_total
        
        # Internal state
        self.models = {}
        self.df_labeled = None
        self.feature_cols = None
        self.global_used_questions = set()
        self.generated_papers = []
        self.exam_depleted = False
        self.textbook_depleted = False

    def load_and_flatten_json(self):
        """
        Load and flatten the questions JSON file into a DataFrame.
        
        Returns
        -------
        pd.DataFrame
            Flattened questions with all relevant features
        """
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

        # Calculate concept recency
        exam_df = df[(df["source"] == "exam") & (df["year"] > 0)]
        concept_last_seen = exam_df.groupby("concept_id")["year"].max().to_dict()
        df["concept_last_seen_year"] = df["concept_id"].map(concept_last_seen).fillna(0)

        print(f"✓ Loaded {len(df)} questions")
        print(f"  Exam: {len(df[df['source'] == 'exam'])} | Textbook: {len(df[df['source'] != 'exam'])}")
        return df

    def create_labels(self, df, use_soft_labels=True, threshold=0.5):
        """
        Create training labels for questions.
        
        Parameters
        ----------
        df : pd.DataFrame
            Questions dataframe
        use_soft_labels : bool, optional
            Use soft labeling for textbook questions (default: True)
        threshold : float, optional
            Probability threshold for soft labels (default: 0.5)
            
        Returns
        -------
        pd.DataFrame
            DataFrame with labels added
        """
        df = df.copy()
        df["label"] = 0

        # Label recent exam questions as positive
        exam_mask = (df["source"] == "exam") & (df["year"] >= self.target_year - 8)
        df.loc[exam_mask, "label"] = 1

        if use_soft_labels:
            # Calculate appearance probability for textbook questions
            tb_mask = df["source"] != "exam"
            df.loc[tb_mask, "appear_probability"] = (
                df.loc[tb_mask, "topic_importance"] * 0.45 +
                df.loc[tb_mask, "concept_importance"] * 0.45 +
                df.loc[tb_mask, "section_confidence"] * 0.10
            )

            # Normalize probabilities
            min_p = df.loc[tb_mask, "appear_probability"].min()
            max_p = df.loc[tb_mask, "appear_probability"].max()
            if max_p > min_p:
                df.loc[tb_mask, "appear_probability"] = (
                    (df.loc[tb_mask, "appear_probability"] - min_p) / (max_p - min_p)
                )

            # Apply threshold
            df.loc[tb_mask, "label"] = (
                df.loc[tb_mask, "appear_probability"] > threshold
            ).astype(int)

        print(f"✓ Created {df['label'].sum()} positive labels")
        return df

    def engineer_features(self, df):
        """
        Engineer features for ML models.
        
        Parameters
        ----------
        df : pd.DataFrame
            Questions dataframe with labels
            
        Returns
        -------
        tuple
            (X, y, feature_cols, data) where:
            - X: Feature matrix
            - y: Labels
            - feature_cols: List of feature column names
            - data: Full dataframe with engineered features
        """
        data = df.copy()
        baseline_year = self.target_year - 4

        # Concept recency features
        data["concept_last_seen_year"] = data["concept_last_seen_year"].replace(0, baseline_year)
        data["concept_years_since_last_seen"] = (
            self.target_year - data["concept_last_seen_year"]
        ).clip(lower=0, upper=10)

        data["concept_recency_decay"] = np.exp(-0.2 * data["concept_years_since_last_seen"])
        data["importance_x_concept_recency"] = (
            data["concept_importance"] * data["concept_recency_decay"]
        )

        # Binary features
        data["is_exam_source"] = (data["source"] == "exam").astype(int)
        data["high_topic_importance"] = (data["topic_importance"] > data["topic_importance"].median()).astype(int)
        data["high_section_conf"] = (data["section_confidence"] > data["section_confidence"].median()).astype(int)
        data["concept_x_section"] = data["concept_importance"] * data["section_confidence"]

        # Section one-hot encoding
        masked_section = data["section"].where(data["source"] == "exam", "UNKNOWN")
        section_dummies = pd.get_dummies(masked_section, prefix="section")
        data = pd.concat([data, section_dummies], axis=1)

        # Define feature columns
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
        """
        Train ensemble of ML models.
        
        Parameters
        ----------
        X_train : pd.DataFrame
            Training features
        y_train : pd.Series
            Training labels
            
        Returns
        -------
        dict
            Dictionary of trained models
        """
        # Random Forest
        rf = RandomForestClassifier(
            n_estimators=100, max_depth=4, min_samples_leaf=8,
            min_samples_split=15, class_weight="balanced",
            random_state=42, n_jobs=-1,
        )

        # XGBoost
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
        """
        Predict appearance probabilities using ensemble.
        
        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix
            
        Returns
        -------
        np.ndarray
            Predicted probabilities for each question
        """
        proba_dict = {name: model.predict_proba(X)[:, 1] 
                     for name, model in self.models.items()}
        ensemble_score = 0.5 * proba_dict["RandomForest"] + 0.5 * proba_dict["XGBoost"]

        # Apply recency boost for exam questions
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
        """
        Train the model and prepare for paper generation.
        
        Returns
        -------
        self
            Returns self for method chaining
        """
        print(f"\n{'='*80}")
        print(f"TRAINING MODEL FOR YEAR {self.target_year}")
        print(f"{'='*80}")
        print(f"Automatic settings:")
        print(f"  - Min score threshold: {self.MIN_SCORE}")
        print(f"  - Max papers allowed: {self.MAX_PAPERS}")
        print(f"{'='*80}\n")
        
        # Load and prepare data
        df_raw = self.load_and_flatten_json()
        self.df_labeled = self.create_labels(df_raw)
        X, y, self.feature_cols, _ = self.engineer_features(self.df_labeled)

        # Create training set
        textbook_mask = self.df_labeled["source"] == "textbook"
        sampled_textbook = textbook_mask & (np.random.rand(len(self.df_labeled)) < 0.2)

        train_mask = (
            ((self.df_labeled["source"] == "exam") &
             (self.df_labeled["year"] < self.target_year - 2)) |
            sampled_textbook
        )

        X_train, y_train = X[train_mask.values], y[train_mask.values]
        
        # Train models
        self.models = self.train_models(X_train, y_train)

        # Predict and filter
        self.df_labeled["prediction_score"] = self.predict_probabilities(X)
        
        before_filter = len(self.df_labeled)
        self.df_labeled = self.df_labeled[
            self.df_labeled["prediction_score"] >= self.MIN_SCORE
        ].copy()
        after_filter = len(self.df_labeled)
        
        print(f"\n✓ Models trained successfully")
        print(f"✓ Filtered to high-quality questions: {after_filter}/{before_filter} "
              f"(removed {before_filter - after_filter})")
        print(f"{'='*80}\n")
        
        return self

    def remove_similar_questions(self, questions_df, threshold=0.75, used_ids=None):
        """
        Remove similar questions using cosine similarity.
        
        Parameters
        ----------
        questions_df : pd.DataFrame
            Questions to deduplicate
        threshold : float, optional
            Similarity threshold (default: 0.75)
        used_ids : set, optional
            IDs of already used questions
            
        Returns
        -------
        pd.DataFrame
            Deduplicated questions
        """
        if len(questions_df) == 0:
            return questions_df

        used_ids = used_ids or set()
        if used_ids:
            questions_df = questions_df[~questions_df["id"].isin(used_ids)].copy()

        if len(questions_df) == 0:
            return questions_df

        # Remove exact duplicates
        questions_df = questions_df.drop_duplicates(subset=["raw_text"], keep="first")

        if "embedding" not in questions_df.columns:
            return questions_df

        valid = questions_df[questions_df["embedding"].notna()].copy()
        if len(valid) == 0:
            return questions_df

        # Calculate similarity matrix
        embeddings = np.array(valid["embedding"].tolist())
        sim = cosine_similarity(embeddings)

        # Remove similar questions
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
    
    def define_paper_structure(self):
        """
        Define the structure of question papers.
    
        Returns
        ------
        dict
            Paper structure specification
        
        Notes
        -----
        You can customize this function to match your exam format.
        Current structure is based on COMP.pdf format.
        """
        return {
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
                "priority_topics": [
                    "structure", "struct", "file", 
                    "storage class", "dynamic memory"
                ],
                "require_multipart": True,
            },
        }
    def generate_question_paper(self, structure, similarity_threshold=0.75,
                                paper_number=1, exam_textbook_ratio=0.3):
        """
        Generate a single question paper.
        
        Parameters
        ----------
        structure : dict
            Paper structure specification
        similarity_threshold : float, optional
            Similarity threshold for deduplication (default: 0.75)
        paper_number : int, optional
            Paper number (default: 1)
        exam_textbook_ratio : float, optional
            Ratio of exam to textbook questions (default: 0.3)
            
        Returns
        -------
        dict
            Generated question paper
        """
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

            # Apply priority topics filter
            if cfg.get("priority_topics"):
                pat = "|".join(cfg["priority_topics"])
                pr = available[available["raw_text"].str.lower().str.contains(pat, na=False)]
                if len(pr) > 0:
                    available = pr

            # Apply multipart requirement
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

            # Apply length constraints
            min_len = cfg.get("min_length", 0)
            max_len = cfg.get("max_length", float("inf"))
            available = available[
                (available["text_length"] >= min_len) &
                (available["text_length"] <= max_len)
            ]

            # Section matching
            sec_match = available[
                (available["section"] == section_name) |
                (available["section"] == "UNKNOWN")
            ]
            section_questions = sec_match if len(sec_match) > 0 else available

            # Remove similar questions
            section_questions = self.remove_similar_questions(
                section_questions, threshold=similarity_threshold, used_ids=all_used
            )

            if len(section_questions) == 0:
                warning = f" No questions available for {section_name}"
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
                    warning = f" EXAM questions depleted! Using textbook questions only."
                    paper["metadata"]["warnings"].append(warning)
                    self.exam_depleted = True
                target_exam = 0
                target_textbook = target_count
            elif actual_textbook_available == 0 and actual_exam_available > 0:
                if not self.textbook_depleted:
                    warning = f" TEXTBOOK questions depleted! Using exam questions only."
                    paper["metadata"]["warnings"].append(warning)
                    self.textbook_depleted = True
                target_exam = target_count
                target_textbook = 0
            elif actual_exam_available == 0 and actual_textbook_available == 0:
                warning = f" CRITICAL: All questions depleted for {section_name}!"
                paper["metadata"]["warnings"].append(warning)
                paper["sections"][section_name] = {
                    "instruction": cfg.get("instruction", ""),
                    "questions": []
                }
                continue
            else:
                # Handle partial depletion
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

            # Fill remaining slots
            while len(selected_dicts) < target_count:
                selected_ids = {q['id'] for q in selected_dicts}
                remaining_textbook = textbook_q[~textbook_q["id"].isin(selected_ids)]
                remaining_exam = exam_q[~exam_q["id"].isin(selected_ids)]

                found = False
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

    def generate_multiple_papers(self, num_papers, structure=None, exam_textbook_ratio=0.3):
        """
        Generate multiple question papers.
        
        Parameters
        ----------
        num_papers : int
            Number of papers to generate
        structure : dict
            Paper structure specification
        exam_textbook_ratio : float, optional
            Ratio of exam to textbook questions (default: 0.3)
            
        Returns
        -------
        list
            List of generated papers
        """

        if structure is None:
            structure = self.define_paper_structure()
        # Validate number of papers
        if num_papers > self.MAX_PAPERS:
            print(f"\n ERROR: Cannot generate more than {self.MAX_PAPERS} papers!")
            print(f"   Requested: {num_papers}")
            print(f"   Maximum allowed: {self.MAX_PAPERS}")
            print(f"\n Tip: Question sets more than {self.MAX_PAPERS} are not available.\n")
            return []
        
        if num_papers < 1:
            print("\n ERROR: Need at least 1 paper!\n")
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
                    print(f" {len(paper['metadata']['warnings'])} warning(s)")
            else:
                print(f" No questions generated!")

        print(f"\n{'='*80}")
        print(f"Successfully generated {len(papers)} paper{'s' if len(papers) > 1 else ''}")
        print(f"{'='*80}\n")

        return papers

    def display_question_paper(self, paper):
        """
        Display a single question paper.
        
        Parameters
        ----------
        paper : dict
            Paper to display
        """
        meta = paper.get("metadata", {})

        print("=" * 80)
        print(f"PAPER #{meta.get('paper_number', '?')} - YEAR {meta.get('target_year', '?')}")
        print(f"Target ratio: {meta.get('exam_textbook_ratio', 0)*100:.0f}% exam")
        print("=" * 80)

        if meta.get('warnings'):
            print("\n  WARNINGS:")
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
        """
        Display all generated papers.
        
        Parameters
        ----------
        papers : list
            List of papers to display
        """
        for i, paper in enumerate(papers, 1):
            self.display_question_paper(paper)
            if i < len(papers):
                print("\n" * 2)

    def get_stats(self):
        """
        Get generation statistics.
        
        Returns
        -------
        dict
            Statistics about generated papers
        """
        return {
            "papers_generated": len(self.generated_papers),
            "questions_used": len(self.global_used_questions),
            "questions_remaining": len(self.df_labeled) - len(self.global_used_questions),
            "target_year": self.target_year,
            "exam_depleted": self.exam_depleted,
            "textbook_depleted": self.textbook_depleted,
        }
