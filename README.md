# Reddit-Flair-Detector

This Reddit Flair Detector is a Flask web application meant to detect flairs of r/India subreddit posts using Machine Learning algorithms. The application is live at [Reddit Flair Detector](https://reddit-unflair.herokuapp.com/).

### Approach

Having worked on sentiment analysis problems before, I had some intuition about how I would train my model. I focused on the text-based features i.e title, url, body, author and comments. Since they all ultimately were text, I decided to try different combinations of these features to train my model. I read about the most commonly used Machine Learning algorithms for text processing, and experimented with 5 algorithms that are known to give the best results.
I obtained test accuracies on various scenarios which have been documented in the Results section.

The step-wise approach taken for the task is as follows:

  1. Initially, I collected 100 r/India subreddit posts for each of the 12 flairs, but the best accuracy I could achieve was around 77%. So, I increased the number to 300 posts per flair. The data was collected using the Reddit API `praw` module [[1]](https://towardsdatascience.com/scraping-reddit-data-1c0af3040768).
  2. The data included *id, title, comments, body, url, author, time-created, number of comments* and *comments* (top 5 comments only).
  3. Pre-processing was performed on the textual data using `nltk`, which included - 
      1) Converting all text to lowercase
      2) `Regex Tokenizer` to remove unnecessary characters
      3) `Porter Stemmer` for stemming 
      4) `Stopword` removal 
      5) Removing non-english words was initially considered using the python library `enchant`, but the idea was discarded due to reduction in accuracy

  4. Three types of features were considered for the the given task:
    
      1) Comments only 
      2) Body + Comments 
      3) Title + Url + Body + Comments

  5. The dataset was split into **80% train** and **20% test** data using `train-test-split` of `scikit-learn`.
  6. A `pipeline` to convert the data into `Vector`, `TF-IDF` forms and classify it, was created using `scikit-learn`.
  7. The following ML algorithms (using `scikit-learn` libraries) were applied on the dataset:
    
      1) Multinomial Naive Bayes
      2) Logistic Regression
      3) Random Forest Classifier
      4) Multilayer Perceptron
      5) Linear Support Vector Machine
    
   Experimented with parameters like 'ngram_range', and classifier specific parameters to obtain best results.

   8. Training and Testing on the dataset showed the **Random Forest Classifier** gave the best testing accuracy of **81.002%** when trained on the combination of **Body + Title + Comments + Url** feature.
   9. This model was saved and used for prediction of the flair from the URL of the post, at the backend of the website.

### Directory Structure

The description of files and folders can be found below:
  
  1. [Data](https://github.com/Anukriti2512/Reddit-Flair-Detector/tree/master/Data) - Folder containing the collected data in .csv and .json formats.
  2. [Jupyter Files](https://github.com/Anukriti2512/Reddit-Flair-Detector/tree/master/Jupyter%20files) - Folder containing Jupyter Notebook to collect Reddit India data and train Machine Learning models.
  3. [MongoDB dump](https://github.com/Anukriti2512/Reddit-Flair-Detector/tree/master/MongoDB%20dump) - Folder containing the MongoDB instances of collected data.
  4. [website](https://github.com/Anukriti2512/Reddit-Flair-Detector/tree/master/website) - Folder containing all website related files and folders:

      1) static/css - CSS code for website
      2) templates - HTML code for website
      3) Procfile - Needed to setup Heroku
      4) [app.py](https://github.com/Anukriti2512/Reddit-Flair-Detector/blob/master/website/app.py) - File containing the main application code which loads the Machine Learning model and renders the results.
      5) requirements.txt - Containing all Python dependencies of the project.
      6) nltk.txt - Containing all NLTK library needed dependencies (required by Heroku)

  5. [model.sav.zip](https://github.com/Anukriti2512/Reddit-Flair-Detector/blob/master/model.sav.zip) - The trained ML model in zipped format
  
### Tech-stack

The entire project has been developed using Python programming language, utilizing it's powerful text processing and machine learning modules. Due to no prior experience with web-development, using Python for the web-application as well was preferred. Hence, web-app was developed using Flask web-framework, which is quite intuitive for beginners like myself. The web-app is hosted on Heroku server.

### Project Execution

  1. Open the `Terminal`.
  2. Clone the repository by entering `git clone https://github.com/Anukriti2512/Reddit-Flair-Detector.git`.
  3. Ensure that `Python3`, `pip` and `Anaconda 3` are installed on the system.
  4. Create a `conda env` by executing the following command: `conda create --name myenv`.
  5. Activate the conda environment by executing the follwing command: `conda activate myenv`.
  6. Unzip the `model.sav.zip` file and move it inside the `website` folder.
  7. Change the working directory to `website` by using the `cd` command.
  8. Execute `pip install -r requirements.txt`.
  9. Enter `python` shell and `import nltk`. Execute `nltk.download('stopwords')` and exit the shell.
  10. Create an account on Reddit, and get your credentials for using the API. Enter the credentials in `detect_flair()` function in the `app.py` file.
  10. Now, execute: `python app.py` and it will point to the `localhost` with the port.
  11. Copy this `IP Address` on a web browser and use the application.
    
### Results

#### Comments only as Feature

| Algorithm                  | Test Accuracy     |
| -------------              |:-----------------:|
| Multinomial Naive Bayes    | 0.5957142857      |
| Logistic Regression        | 0.6038923890      |
| Random Forest              | **0.7039278095**  |
| MLP                        | 0.4011387822      |
| Linear SVM                 | 0.6289874762      |

#### Body + Comments as Feature

| Algorithm                  | Test Accuracy     |
| -------------              |:-----------------:|
| Multinomial Naive Bayes    | 0.6151504832      |
| Logistic Regression        | 0.6283922095      |
| Random Forest              | **0.7539285714**  |
| MLP                        | 0.4760714286      |
| Linear SVM                 | 0.6830438095      |

#### Title + Comments + URL + Body as Feature

| Algorithm                  | Test Accuracy     |
| -------------              |:-----------------:|
| Multinomial Naive Bayes    | 0.636743215031    |
| Logistic Regression        | 0.747390396659    |
| Random Forest              | **0.8100208768**  |
| MLP                        | 0.519832985386    |
| Linear SVM                 | 0.7244258872651   |

### Under construction

1. The UI is still being improved.
2. Currently, only correct Reddit India URLs will work. Cases for invalid URLs need to be handled.
3. Working on a page to display temporal and user analysis visualizations.

### References

1. [Scraping data from Reddit](https://towardsdatascience.com/scraping-reddit-data-1c0af3040768)
2. [PRAW Documentation](https://praw.readthedocs.io/en/latest/)
3. [Creating web-application using Flask](https://towardsdatascience.com/designing-a-machine-learning-model-and-deploying-it-using-flask-on-heroku-9558ce6bde7b)
4. [HTML+CSS](https://towardsdatascience.com/develop-a-nlp-model-in-python-deploy-it-with-flask-step-by-step-744f3bdd7776)
5. [Readme template](https://gist.github.com/PurpleBooth/109311bb0361f32d87a2)
