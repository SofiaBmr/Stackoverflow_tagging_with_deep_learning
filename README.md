# Machine Learning in production : stackoverflow question classification

---

This project aims to construct a model that automatically tag stackoverflow.com questions by using their title. For this multi-class classification task, we will use a Keras sequential model combined with a pre-trained BERT model to embed the input.

The main objective was to build a production-ready ML project. This is why the performances of the model are not exceptional. We've paid attention to :
- the organisation and the versioning 
- the reproducibility and the portability
- the use of tests
# Launch the code

---

Run the following command on your terminal : 
```bash
flask run
```
Once the application is launched, a link will be generated, please open it in your search engine.

If you have any issue with the module, please try first in your terminal (replacing with your own project's location) :

```bash
export PYTHONPATH=$PYTHONPATH:/Users/poc-to-prod-capstone
```

Several versions of the model are available in the folder **/train/data/artefacts/models**. If you want to use a different one, please change the folder names in the code. It is also possible to train a completely new model, with new parameters by changing the train-conf.yml file and re-traning the model.

# Author
Sofia Boumahrat P2023

Special mention to Gabriel Truong & Mathieu Laversin who helped me in this hardship.
