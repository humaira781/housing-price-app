if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 10000)))
    from flask import Flask, render_template, request
   




import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Feature 1: CRIM
    crim = float(request.form['crim'])

    # Feature 2: ZN
    zn = float(request.form['zn'])

    # Feature 3: INDUS
    indus = float(request.form['indus'])

    # Feature 4: CHAS
    chas = float(request.form['chas'])

    # Feature 5: NOX
    nox = float(request.form['nox'])

    # Feature 6: RM
    rm = float(request.form['rm'])

    # Feature 7: AGE
    age = float(request.form['age'])

    # Feature 8: DIS
    dis = float(request.form['dis'])

    # Feature 9: RAD
    rad = float(request.form['rad'])

    # Feature 10: TAX
    tax = float(request.form['tax'])

    # Feature 11: PTRATIO
    ptratio = float(request.form['ptratio'])

    # Feature 12: B
    b = float(request.form['b'])

    # Feature 13: LSTAT
    lstat = float(request.form['lstat'])

    # Combine all features into a single array
    features = np.array([[crim, zn, indus, chas, nox, rm, age, dis, rad, tax, ptratio, b, lstat]])

    prediction = model.predict(features)
    return render_template('index.html', prediction_text="Predicted House Price: ${:.2f}".format(prediction[0]))

if __name__ == "__main__":
    app.run(debug=True)
