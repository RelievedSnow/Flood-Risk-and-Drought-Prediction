from flask import Flask, render_template, request, url_for
import numpy as np
from joblib import load, dump
from rainfall_prediction import accuracy, rainfall_data
import matplotlib.pyplot as plt
import seaborn as sns
import os

app = Flask(__name__)

@app.route('/', methods=['POST','GET'])
def home():
    if request.method == 'GET':
        return render_template("index.html", href='static/images/drought-and-flood.jpg', avatar='static/images/avatar.jpg')
    else:
        # Get form data from the request
        June = request.form['june']
        July = request.form['july']
        August = request.form['august']
        September = request.form['september']

        # Convert inputs to numpy array
        input_data = [June, July, August, September]
        input_data = np.array(input_data, dtype=float).reshape(1, -1)

        # Load the model
        clf = load('rainfall_prediction_model.joblib')

        # Make prediction
        prediction = clf.predict(input_data)

        # Extract predicted value (assuming it's dissolved oxygen, DO)
        predicted_value = prediction[0]
   
        accuracy_score = accuracy

        makePicture('RS_Session_259_AU_1203_1.csv', clf, input_data, 'rainfall_prediction_bar_plot.png')

        # Render result template with the prediction and the plot URL)
        plot_url = url_for('static', filename='images/rainfall_prediction_bar_plot.png')
        return render_template("result.html", predicted_value=predicted_value, accuracy_score=accuracy_score, plot_url=plot_url,)


def makePicture(traning_data_file, model, input_data, outputfile_1):
    # Create a color palette for different conditions
    palette = {'Flood': 'red', 'Drought': 'blue', 'Normal': 'green'}

    # Set a smaller figure size
    plt.figure(figsize=(12, 4))

    # Plot the data as a bar plot
    sns.barplot(x='YEAR', y='JUN-SEP', hue='Condition', data=rainfall_data, palette=palette)

    # Add titles and labels
    plt.title('Flood, Drought, and Normal Conditions Over the Years', fontsize=14)
    plt.xlabel('Year', fontsize=10)
    plt.ylabel('Total Rainfall (June - September)', fontsize=10)
    plt.xticks(rotation=90, fontsize=10)  # Rotate x-axis labels and reduce font size for better readability
    plt.yticks(fontsize=10)
    plt.legend(title='Condition', fontsize=10)

    # Adjust layout for a smaller plot
    plt.tight_layout()
    # Save the plot as an image in the static folder
    plot_path = os.path.join('static', 'images', 'rainfall_prediction_bar_plot.png')
    plt.savefig(plot_path)
    plt.close()

if __name__ == '__main__':
    app.run(debug=True)
