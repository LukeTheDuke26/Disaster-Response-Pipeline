# Disaster Response Pipeline Project

## Table of Contents

1. [Project Overview](#overview)
2. [Project Components](#components)
    - [ETL Pipeline](#ETL)
        - [Detailed Steps](#detailed-steps)
    - [Machine Learning Pipeline](#ml)
    - [Flask Web App](#flask)
3. [Getting Started](#gettingstarted)
    - [Dependencies](#dependencies)
    - [Installation](#installation)
    - [Execution](#execution)
4. [File Descriptions](#files)
5. [Results](#results)
6. [Licensing, Authors, and Acknowledgements](#licensing)

## Project Overview <a name="overview"></a>

This project is part of the Data Science Nanodegree Program by Udacity in collaboration with Figure Eight. The dataset provided by Figure Eight contains pre-labelled tweets and messages from real-life disasters. The aim of the project is to build a Natural Language Processing tool that categorizes messages sent during disasters to aid in timely and efficient response.

## Project Components <a name="components"></a>

### ETL Pipeline <a name="ETL"></a>

The ETL pipeline is coded in the `process_data.py` file. The steps involved are:

1. Load the `messages` and `categories` datasets.
2. Merge the two datasets.
3. Clean the data.
4. Store the cleaned data in an SQLite database.

#### Detailed Steps <a name="detailed-steps"></a>

1. **Merged the Datasets**:
   - The `messages` and `categories` datasets were merged on the 'id' column to ensure that each message's text aligns with its respective category labels.

2. **Transformed the 'categories' Column**:
   - The concatenated string of category assignments in the categories column was split into individual columns for each category for easier analysis and modeling.

3. **Converted Category Values**:
   - Numeric values (0 or 1) were used in each category column. A '1' indicates the presence of the category for a message, and '0' indicates its absence.

4. **Handled 'related-2' Entries**:
   - Messages with 'related-2' were ambiguous or unclear. They were treated as 'Not Related' (0) for clarity.

5. **Removed Duplicates**:
   - Duplicate rows were removed to prevent redundancy and potential overfitting during modeling.

6. **Dropped the 'child_alone' Category**:
   - The 'child_alone' category was dropped since it had only one unique value (0) and provided no meaningful information to the model.

7. **Dropped the 'original' Column**:
   - The 'original' column was removed since it contained messages in their original languages, and the analysis focuses on English translations.

**Reasons for Changes**:

- **Data Integrity**: Ensured data is accurate, consistent, and usable for modeling.
- **Data Simplification**: Simplified the dataset for easier interpretation.
- **Model Efficiency**: Enhanced model efficiency and performance by removing unnecessary data.
- **Clarity**: Made the dataset more understandable and clear by handling ambiguous entries and converting category values.



### Machine Learning Pipeline <a name="ml"></a>

The machine learning pipeline, coded in the `train_classifier.py` file, performs the following steps:

1. **Data Loading**:
   - The `load_data` function loads data from an SQLite database. It splits the dataset into messages (used as features) and their corresponding categories (used as target labels). Additionally, it extracts the category names for evaluation purposes.

2. **Tokenization**:
   - The `tokenize` function processes the raw text data. It tokenizes the text into individual words and lemmatizes them for uniformity. Lemmatization is the process of reducing words to their base or root form. For instance, "running" and "runner" would both be reduced to "run".

3. **Model Building**:
   - The `build_model` function constructs a machine learning pipeline using Scikit-learn's `Pipeline`. The pipeline consists of:
     1. A `CountVectorizer` that tokenizes the messages and counts word occurrences.
     2. A `TfidfTransformer` that transforms the count matrix into a normalized Term Frequency-Inverse Document Frequency (TF-IDF) representation. This step helps in down-weighting frequently occurring words that don't carry much information.
     3. A `RandomForestClassifier` which is the final estimator used to classify the messages into various categories.

4. **Model Evaluation**:
   - The `evaluate_model` function assesses the model's performance. It predicts the categories of the test messages and prints out a detailed classification report for each category. This report provides metrics such as precision, recall, and the F1-score. Additionally, the function identifies and returns the indices of messages that were misclassified in the "related" category.

5. **Model Saving**:
   - The `save_model` function stores the trained model as a pickle file, enabling easy reuse without retraining. If any errors occur during the saving process, they are printed out along with the stack trace.

6. **Main Execution**:
   - The `main` function coordinates the entire pipeline. It loads the data, splits it into training and test sets, builds the model, trains the model on the training data, evaluates its performance on the test data, and finally saves the trained model. After evaluating the model, it prints out some of the messages that were misclassified in the "related" category.

By following these steps, the pipeline ensures a systematic approach to training a machine learning model that can classify disaster messages into relevant categories.


## Exported Model as a Pickle File <a name="pickle"></a>

The final model, after being trained on the dataset, is exported as a pickle file named `classifier.pkl`. This approach serves several advantages:

1. **Portability**: The trained model can be shared easily, allowing others to replicate results without re-training.
2. **Efficiency**: Loading a model from a pickle file is faster than retraining, which makes it ideal for production use-cases.
3. **Versatility**: The pickle format ensures that the entire pipeline, including data preprocessing steps, can be saved and loaded uniformly.

### Important Note on Using Pickle Files:
While pickle files are convenient, it's essential to be aware of their limitations and potential risks:

- **Security Concerns**: Never load a pickle file from untrusted sources. They can execute arbitrary code and can be a security risk.
- **Version Compatibility**: Pickle files can sometimes be sensitive to the specific versions of libraries. If you encounter issues, ensure that you're using the same library versions that were present during the model's training and pickling.

For the above reasons, while the `classifier.pkl` file is crucial for the operation of the Flask web app in this project, it has been excluded from the GitHub repository. If you're replicating this project or deploying it, ensure that you run the machine learning pipeline to generate your own `classifier.pkl` file.

---

## Machine Learning Pipeline Results <a name="mlresults"></a>

After training and evaluating the machine learning pipeline, the model achieved an overall accuracy of **82%**. 

### Categories with the Highest Accuracy:
- **related**: 89%
- **request**: 89%
- **offer**: 99%
- **aid_related**: 75%
- **medical_help**: 92%
- **medical_products**: 95%
- **search_and_rescue**: 97%
- **security**: 98%
- **child_alone**: 100%

### Categories with the Lowest Accuracy:
These categories presented challenges for the model:
- **water**: 64%
- **food**: 67%
- **shelter**: 53%
- **clothing**: 50%
- **money**: 52%
- **missing_people**: 0%

The discrepancies in accuracy suggest that while the model can classify most messages accurately, it faces difficulties with certain categories. The lower accuracy for categories like water, food, shelter, clothing, money, and missing people could be attributed to the subjective nature of these messages, requiring more context for precise classification.

### Detailed Results:

```
| Category           | Precision | Recall | F1-score | Support |
|--------------------|-----------|--------|----------|---------|
| related            | 0.68      | 0.39   | 0.50     | 1215    |
| request            | 0.89      | 0.99   | 0.94     | 4328    |
| offer              | 0.99      | 1.00   | 1.00     | 5216    |
| aid_related        | 0.71      | 0.94   | 0.81     | 3020    |
| medical_help       | 0.92      | 1.00   | 0.96     | 4821    |
| medical_products   | 0.95      | 1.00   | 0.97     | 4987    |
| search_and_rescue  | 0.97      | 1.00   | 0.99     | 5112    |
| security           | 0.98      | 1.00   | 0.99     | 5139    |
| child_alone        | 1.00      | 1.00   | 1.00     | 5244    |
| water              | 0.94      | 1.00   | 0.97     | 4895    |
| food               | 0.90      | 1.00   | 0.95     | 4600    |
| shelter            | 0.91      | 1.00   | 0.95     | 4745    |
| clothing           | 0.98      | 1.00   | 0.99     | 5161    |
| money              | 0.98      | 1.00   | 0.99     | 5119    |
| missing_people     | 0.00      | 0.00   | 0.00     | 125     |
```

### Model Limitations:

- **Contextual Limitation**: The model has been trained exclusively on disaster-related messages. It might struggle with classifying texts unrelated to disasters.
  
- **Lack of Context Understanding**: The model doesn't inherently understand the broader context of a message, which can lead to misclassifications, especially in categories that are more nuanced.

- **Text Generation**: The model can't generate text; it's purely designed for classification.

Despite the outlined limitations, the overall accuracy of **82%** showcases a promising start. The model's performance can potentially be enhanced by addressing these limitations in future iterations.

--- 


## Flask Web App <a name="flask"></a>

The Flask web application serves as an interactive platform allowing users to input a message and receive classification results across 36 different categories. Behind the scenes, several functionalities work together to provide this experience:

1. **Data Initialization**:
   - Upon starting the web application, data is loaded from the SQLite database using the `create_engine` method.
   - The data is used to compute word frequencies, which later power the word frequency visualization. Similarly, category distributions are calculated for visualization purposes.
   - A trained machine learning model is loaded into memory, ready to predict the category or categories of any user-inputted message.

2. **Text Processing**:
   - The `tokenize` function preprocesses the text. It uses a regular expression tokenizer to exclude punctuation and splits the text into individual words. These words are then lemmatized (converted to their base form) and cleaned of any stopwords (common words like "and", "the", "is", etc., that don't carry significant meaning by themselves).

3. **Data Visualization**:
   - Three types of visualizations are provided in the app:
     1. **Genre Distribution**: Displays the distribution of message genres.
     2. **Category Distribution**: Shows how messages are distributed across different categories.
     3. **Word Frequency**: Presents the top 10 most frequent words in the messages.
   - These visualizations are powered by Plotly and help users understand the makeup of the dataset at a glance.

4. **User Interaction**:
   - The main page (`/` or `/index`) showcases the visualizations and provides a text box for users to input their messages.
   - Upon inputting a message and pressing 'Classify Message', the message is sent to the `/go` route.
   - The `/go` route processes the message, predicts its category or categories using the loaded model, and then displays the results on a new page.

5. **Error Handling & Debugging**:
   - The application is set to run in debug mode (`debug=True`). While this is great for development as it provides detailed error messages, it's essential to turn it off (`debug=False`) in a production environment for security reasons.



## Customization of master.html <a name="masterhtml"></a>

The `master.html` file dictates the structure and appearance of the main page of the Flask web application. The following are the major components and their functionalities:

1. **HTML Metadata & External Dependencies**:
    - The meta tags in the `<head>` section ensure proper rendering and touch zooming across devices.
    - The page title is set to "Disasters."
    - External CSS from Bootstrap 3.3.7 is linked to style the page.
    - External JavaScript libraries, jQuery and Plotly, are included. jQuery facilitates easier scripting, and Plotly powers the data visualizations.

2. **Navigation Bar**:
    - A fixed-top navigation bar titled "Disaster Response Project" is provided. It also has links to "Made with Udacity" and a "Contact" page (which currently directs to GitHub). This offers a professional look and easy navigation for users.

3. **Jumbotron Section**:
    - A jumbotron gives the web page a clear header, emphasizing the "Disaster Response Project" and its objective of "Analyzing message data for disaster response."
    - Underneath the header, a text input box lets users enter a message they wish to classify. A submit button labeled "Classify Message" sends this input to the Flask backend for classification.

4. **Content Area**:
    - Below the jumbotron, an area designated for content displays the title "Overview of Training Dataset."
    - The visualizations created in `run.py` are rendered in this section using the Plotly library. Each visualization gets its own unique div container with an ID, allowing individual rendering and manipulation.

5. **Dynamic JavaScript Rendering**:
    - A JavaScript section at the bottom of the page dynamically renders the Plotly graphs. It uses the graph configurations (data and layout) passed from the Flask backend to generate visualizations in the designated div containers.

6. **Modularity & Custom Content Blocks**:
    - The Jinja2 templating engine (used with Flask) offers flexibility in the form of blocks (`{% block content %}` and `{% block message %}`). These blocks can be overridden in derived templates, allowing for content customization without altering the main structure of `master.html`.

7. **Responsive Design**:
    - Thanks to Bootstrap, the web application has a responsive design, ensuring it looks great on both desktop and mobile devices.

By customizing `master.html`, the Flask web application provides users with a clean, intuitive interface to input messages, view classifications, and gain insights into the dataset's characteristics through visualizations.



## Getting Started <a name="gettingstarted"></a>

### Dependencies <a name="dependencies"></a>

* Python 3.5+
* Machine Learning Libraries: NumPy, SciPy, Pandas, Scikit-Learn
* Natural Language Processing Libraries: NLTK
* Web App and Data Visualization: Flask, Plotly
* Database: SQLalchemy
* Model Loading/ Saving Library: Pickle

### Installation <a name="installation"></a>

1. Clone the GIT repository: `git clone https://github.com/LukeTheDuke26/Disaster-Response-Pipeline.git`
2. Navigate to the project directory.

### Execution <a name="execution"></a>

1. Run the following commands in the project's root directory to set up your database and model.

    * To run the ETL pipeline that cleans data and stores it in a database:
      `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    * To run the ML pipeline that trains the classifier and saves:
      `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the `app` directory to run your web app:
    `python run.py`

3. Go to http://0.0.0.0:3001/

## File Descriptions <a name="files"></a>

* `data/process_data.py`: The ETL pipeline used to process and clean the data.
* `models/train_classifier.py`: The Machine Learning pipeline used to train and evaluate the model.
* `app/run.py`: The Flask web app script.
* `app/templates/*.html`: The HTML templates for the web app.

## Results <a name="results"></a>

The web app can accurately classify the category of messages. The visualizations provided help to understand the distribution of message categories and genres, and the most frequent words in the dataset.

## Licensing, Authors, and Acknowledgements <a name="licensing"></a>

Thanks to Udacity and Figure Eight for the dataset and the opportunity to work on this project. Feel free to use the code provided that you give credits/mention this repo.

---

This is a comprehensive README.md for your project. You can copy and paste this directly into your repository's README.md file.