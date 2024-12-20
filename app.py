from flask import Flask, render_template, redirect, url_for, request, session
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from currency_converter import CurrencyConverter
import numpy as np
import os


app = Flask(__name__, static_folder='static', template_folder='templates')

app.secret_key = '65jt9e$0r9353W%4w5se9'
# Predefined list of currencies
currencies = ['USD', 'EUR', 'JPY', 'GBP', 'AUD', 'CAD', 'CHF', 'CNY', 'SEK', 'NZD', 'KRW', 'SGD', 'NOK', 'MXN', 'INR',
              'RUB', 'ZAR', 'HKD', 'BRL', 'TRY', 'TWD', 'DKK', 'PLN', 'THB', 'IDR', 'HUF', 'CZK', 'ILS', 'PHP', 'AED',
              'COP', 'SAR', 'MYR', 'VND', 'ARS', 'HRK', 'QAR', 'PKR', 'EGP', 'KZT', 'OMR', 'BDT', 'LKR', 'TND', 'BYN',
              'AZN', 'JOD', 'MAD', 'GTQ', 'BOB', 'AFN', 'NPR', 'PYG', 'MZN', 'BHD', 'XOF', 'SDG', 'BND', 'LYD', 'GHS',
              'BWP', 'NAD', 'SZL', 'GNF', 'SCR', 'GEL', 'TMT', 'ERN', 'BGN', 'FJD', 'JMD', 'MKD', 'NIO', 'PAB', 'RSD',
              'UAH', 'XPF', 'VUV', 'WST', 'CVE', 'STD', 'KGS', 'KES', 'MVR', 'RWF', 'LAK', 'MWK', 'PGK', 'FKP', 'SHP',
              'KYD', 'BBD', 'BMD', 'BSD', 'BZD', 'GYD', 'LRD', 'SBD', 'SRD', 'TOP', 'ZWL', 'BTN', 'IMP', 'CUC', 'CUP',
              'GGP', 'JEP', 'SSP']


@app.route('/converter')
def converter():
    return render_template('converter.html', currencies=currencies)


@app.route('/convert', methods=['POST'])
def convert():
    from_currency = request.form['from_currency']
    to_currency = request.form['to_currency']
    amount = float(request.form['amount'])

    converter = CurrencyConverter()
    try:
        result = round(converter.convert(amount, from_currency, to_currency), 2)
        return render_template('converter.html', result=result, from_currency=from_currency, to_currency=to_currency,
                               amount=amount, currencies=currencies)
    except ValueError:
        error = "Currency conversion failed. Please check your input."
        return render_template('converter.html', error=error, currencies=currencies)


# Placeholder for user authentication (replace with your authentication logic)
def authenticate_user(username, password):
    return username == 'admin' and password == 'admin'


# Load currency detection models
model_500 = load_model('Models/model_500.h5')
model_200 = load_model('Models/model_200.h5')
model_100 = load_model('Models/model_100.h5')


# Function to detect currency using the specified model
def detect_currency(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.

    predictions = model.predict(img_array)
    return predictions[0][0] < 0.5  # Assuming < 0.5 is considered as fake currency


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/login')
def login():
    return render_template('login.html')


@app.route('/logout')
def logout():
    session.clear()  # Clear the session to log out the user
    return redirect(url_for('index'))  # Redirect to the login page or any other page


@app.route('/analysis')
def analysis():
    return render_template('analysis.html')


@app.route('/authenticate', methods=['POST'])
def authenticate():
    username = request.form['username']
    password = request.form['password']

    if authenticate_user(username, password):
        # Redirect to the dashboard route upon successful authentication
        return redirect(url_for('dashboard'))
    else:
        # Display error message on the login page for invalid credentials
        return render_template('login.html', message='Invalid credentials')


@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')


@app.route('/detect', methods=['GET', 'POST'])
def detect():
    if request.method == 'POST':
        file = request.files['file']
        denomination = request.form['denomination']

        if file and file.filename != '':
            upload_dir = 'uploads'
            if not os.path.exists(upload_dir):
                os.makedirs(upload_dir)
            file_path = os.path.join(upload_dir, file.filename)
            file.save(file_path)

            # Determine which model to use based on selected denomination
            if denomination == '500':
                model = model_500
            elif denomination == '200':
                model = model_200
            elif denomination == '100':
                model = model_100
            else:
                return render_template('detect.html', message='Invalid denomination selected')

            # Perform currency detection
            is_fake = detect_currency(file_path, model)
            # os.remove(file_path)

            # Determine prediction result
            result = 'FAKE CURRENCY' if is_fake else 'REAL CURRENCY'

            # Determine which audio file to play based on prediction result
            if is_fake:
                audio_file = 'audio/fake.mp3'
            else:
                audio_file = 'audio/real.mp3'

            # Render the detect template with prediction result
            result_class = result.lower().replace(' ', '-')
            return render_template('detect.html', result=result, result_class=result_class, audio_file=audio_file)

    # Render the detect template if no form submission or invalid request method
    return render_template('detect.html')


if __name__ == '__main__':
    app.run(debug=True)
