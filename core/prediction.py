import numpy as np

# Fast = batch prediction
# Slow = sequential prediction
def predict_eod_with_uncertainty(model, scaler, input_sequence, n_simulations=30, mode='fast'):
    if mode == 'fast':
        input_seq_reshaped = input_sequence.reshape(1, *input_sequence.shape)
        input_seq_batch = np.repeat(input_seq_reshaped, n_simulations, axis=0)  # shape: (30, lookback, features)

        predictions_scaled = model(input_seq_batch, training=True).numpy()
        predictions = scaler.inverse_transform(predictions_scaled)

        mean_pred = predictions.mean()
        std_pred = predictions.std()

        return mean_pred, std_pred

    elif mode == 'slow':
        preds = []
        input_seq_reshaped = input_sequence.reshape(1, *input_sequence.shape)
        for _ in range(n_simulations):
            predicted_scaled = model(input_seq_reshaped, training=True)
            predicted = scaler.inverse_transform(predicted_scaled)[0][0]
            preds.append(predicted)
        return np.mean(preds), np.std(preds)
