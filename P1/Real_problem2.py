import perceptron
import adaline
import reader
import os

if __name__ == '__main__':
    xTrain, yTrain, xPredict, yPredict = reader.leer3("NEURO-prac/P1/Data/problema_real2.txt", "NEURO-prac/P1/Data/problema_real2_no_etiquetados.txt")

    redPerceptron = perceptron.new_perceptron(8, 2, 0.3)
    redAdaline = adaline.new_adaline(8, 2)

    redPerceptron.fit(xTrain, yTrain, epochs=20, alpha=1, verbose=False, ecm=False)
    redAdaline.fit(xTrain, yTrain, epochs=20, alpha=0.1, verbose=False, ecm=False)

    perceptronPredictions = redPerceptron.predict(xPredict)
    adalinePredictions = redAdaline.predict(xPredict)

    predictions_folder = "NEURO-prac/P1/Data/predicciones"
    if not os.path.exists(predictions_folder):
        os.makedirs(predictions_folder)

    perceptron_file_path = os.path.join(predictions_folder, "prediccion_perceptron.txt")
    adaline_file_path = os.path.join(predictions_folder, "prediccion_adaline.txt")

    with open(perceptron_file_path, "w") as f:
        for prediction in perceptronPredictions:
            f.write(f"{prediction[0]}, {prediction[1]}\n")

    with open(adaline_file_path, "w") as f:
        for prediction in adalinePredictions:
            f.write(f"{prediction[0]}, {prediction[1]}\n")