from data.fetch_data import get_btc_data
from data.indicators import add_indicators
from features.process_data import create_features_and_labels
from models.train_model import train_svm_model
from models.predict_model import predict_and_plot

def main():
    btc=get_btc_data()
    btc=add_indicators(btc)
    X_train,X_test,y_train,y_test,btc=create_features_and_labels(btc)
    model=train_svm_model(X_train,y_train)
    predict_and_plot(model,X_test,y_test,btc)

    if __name__=="_main_":
        main()