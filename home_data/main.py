from data import train_data, train_targets, test_data, test_targets
from model_architecture import model, callbacks
import pandas as pd
import numpy as np

if __name__ == "__main__":
    def main():
        history = model.fit(train_data, train_targets, epochs=1000, batch_size=64, validation_split=0.2, callbacks=callbacks)
        model.load_weights('home_data\\best_model\\best_model_homes.keras')
        test_loss, test_acc = model.evaluate(test_data, test_targets)
        print(f"Test acc: {test_acc:.4f}, Test loss: {test_loss:.4f}")
        predictions = np.expm1(model.predict(test_data))

        result = pd.DataFrame({
            'Id': pd.read_csv("home_data\\data\\sample_submission.csv")['Id'],
            'SalePrice': predictions.flatten()
        })
        result.to_csv("home_data\\data\\submission.csv", index=False)
        return 0
    main()