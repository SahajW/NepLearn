from final_model import QuestionPaperGenerator
import pickle

def train_model(year,sets):
    predictor = QuestionPaperGenerator("final_data.json",target_year=year,max_papers=sets)

    # IMPORTANT: must call fit()
    predictor.fit()
    # SAVE THE MODEL OBJECT (NOT PAPER)
    with open("question_predictor.pkl", "wb") as f:
        pickle.dump(predictor, f)

    print("QuestionPaperPredictor saved correctly")

if __name__ == "__main__":
    train_model(2024)