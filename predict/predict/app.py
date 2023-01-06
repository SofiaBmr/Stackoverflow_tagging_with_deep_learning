from flask import Flask, render_template, request, render_template_string
from run import TextPredictionModel

app = Flask(__name__)

@app.route("/", methods=['POST', 'GET'])
def index():
    #return render_template_string("Hello from flask")
    if request.method == "POST":
        question=request.form["question"]
        print(question)
        path = "/Users/boumahrat/Desktop/EPF/5A/from_poc_to_prod/poc-to-prod-capstone/train/data/artefacts/models/2023-01-06-17-25-13"
        prediction_object = TextPredictionModel.from_artefacts(path)

        predic = prediction_object.predict([question], top_k=1)
        print(predic)
        return (str(predic))


    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)