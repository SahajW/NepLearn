from final_model import QuestionPaperPredictor
import pickle

predictor = QuestionPaperPredictor("final_data.json", target_year=2025)

# IMPORTANT: must call fit()
predictor.fit()

# SAVE THE MODEL OBJECT (NOT PAPER)
with open("backend/question_predictor.pkl", "wb") as f:
    pickle.dump(predictor, f)

print("QuestionPaperPredictor saved correctly")
