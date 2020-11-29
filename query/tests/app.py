from flask import Flask, render_template, request

app = Flask(__name__)


@app.route('/', methods=['GET'])
def show_index_html():
    return render_template('index.html')


@app.route('/send_data', methods=['POST'])
def get_data_from_html():
    pay = request.form['pay']
    print("A consuta é: " + pay)
    return "Dado recebido. Por favor, confira o log do programa"


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
