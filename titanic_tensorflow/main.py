from data import train_data, train_targets, test_data, test_targets
from model_architecture import model, callbacks
import pandas as pd
import numpy as np

if __name__ == "__main__":
    def main():
        history = model.fit(train_data, train_targets, epochs=200, batch_size=64, validation_split=0.2, callbacks=callbacks)
        model.load_weights('titanic_tensorflow\\best_model\\best_model_titanic.keras')
        test_loss, test_acc = model.evaluate(test_data, test_targets)
        print(f"Test acc: {test_acc:.4f}, Test loss: {test_loss:.4f}")
        predictions = model.predict(test_data)

        result = pd.DataFrame({
            'PassengerId': pd.read_csv("titanic_tensorflow\\data\\test.csv")['PassengerId'],
            'Survived': (predictions.flatten() > 0.5).astype(int)
        })
        result.to_csv("titanic_tensorflow\\data\\submission.csv", index=False)
        return 0
    main()