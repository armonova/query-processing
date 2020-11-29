from flask import Flask, render_template, request

app = Flask(__name__)


@app.route('/', methods=['GET'])
def show_index_html():
    return render_template('index.html')


@app.route('/send_data', methods=['POST'])
def get_data_from_html():
    pay = request.form['pay']
    model = request.form['model']
    relevant_doc = request.form['relevant_doc']
    print("A consuta é: " + pay)
    print("A modelagem é: " + model)
    print("A arquivo relevante é: " + relevant_doc)
    return "Dado recebido. Por favor, confira o log do programa"


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5001)
