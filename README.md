# mlMushroomDemo

Study project for creating ML demo based on Mushroom Classification kaggle dataset (https://www.kaggle.com/uciml/mushroom-classification).

For this demo was built simple ML model based on CatBoostClassifier.
In this model picked 6 features and make prediction about toxicity mushrooms.
Model have very high value of ROC_AUC quality (0.9999).
The reason of that high value probably is that model is quite simple and has a lot of data.

Demo done using the Streamlit library allows you to select 5 attributes and make prediction for all possible variant of last attribute.  
Demonstration also allows you to select the boundary of the true value that mushroom is poisonous.
