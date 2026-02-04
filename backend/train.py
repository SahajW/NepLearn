from final_model import QuestionPaperGenerator
import pickle

def train_model(year,pastpaper,textbook,sets):
    predictor = QuestionPaperGenerator("final_data.json", target_year=year,target_pastpaper = pastpaper, target_textbook=textbook,target_sets=sets)

    # IMPORTANT: must call fit()
    predictor.fit()
    # SAVE THE MODEL OBJECT (NOT PAPER)
    with open("question_predictor.pkl", "wb") as f:
        pickle.dump(predictor, f)

    print("QuestionPaperPredictor saved correctly")

if __name__ == "__main__":
    train_model(2024)