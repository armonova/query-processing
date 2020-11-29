from flask import Flask, render_template, request
from query.processing import QueryRunner, VectorRankingModel, IndexPreComputedVals

app = Flask(__name__)
index, precomp, cleaner = None, None, None

@app.route('/', methods=['GET'])
def show_index_html():
    return render_template('index.html')


@app.route('/send_data', methods=['POST'])
def get_data_from_html():
    # pay = request.form['pay']
    # model = request.form['model']
    # relevant_doc = request.form['relevant_doc']
    pay = "vocês"
    model = "1"
    relevant_doc = "sao_paulo"

    rank_model = None
    if model == "1":
        rank_model = VectorRankingModel(precomp)
    elif model == "2":
        rank_model = BooleanRankingModel(OPERATOR.AND)
    else:
        rank_model = BooleanRankingModel(OPERATOR.OR)

    print("A consuta é: " + pay)
    print("A modelagem é: " + model)
    print("A arquivo relevante é: " + relevant_doc)

    QueryRunner.runQuery(pay, index, cleaner, rank_model, relevant_doc)

    return "Dado recebido. Por favor, confira o log do programa"

if __name__ == '__main__':
    index, precomp, cleaner = QueryRunner.main()
    app.run(host="0.0.0.0", port=5001)
