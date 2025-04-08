# Topic Modeling with LDA on Reviews

## Overview

This project implements topic modeling using Latent Dirichlet Allocation (LDA) to analyze reviews. The goal is to extract meaningful topics from a collection of text reviews, visualize the results, and save the findings for further analysis.

## Libraries Used

- **Pandas**: For data manipulation and analysis.
- **Matplotlib**: For data visualization.
- **WordCloud**: To generate word clouds from text data.
- **NLTK**: Natural Language Toolkit for text processing.
- **Gensim**: For topic modeling and handling large text corpora.
- **pyLDAvis**: For visualizing LDA models.

## Installation

To install the required libraries, run the following command:

```bash
!pip install pandas matplotlib wordcloud nltk gensim pyLDAvis
```

## Steps in the Project

1. **Import Necessary Libraries**: Load all required libraries for data processing and modeling.

   ```python
   import pandas as pd
   import matplotlib.pyplot as plt
   from wordcloud import WordCloud
   import nltk
   from nltk.corpus import stopwords
   from gensim import corpora
   from gensim.models import LdaModel
   from gensim.models.coherencemodel import CoherenceModel
   import pyLDAvis
   import pyLDAvis.gensim_models as gensimvis
   ```

2. **Upload File with Reviews**: Prompt the user to upload a CSV file containing reviews.

   ```python
   file_name = input("Enter filename with reviews. It should have one column named 'review'")
   df = pd.read_csv(file_name)
   df.dropna(subset=['review'], inplace=True)
   ```

3. **Data Cleaning**:
   - Convert text to lowercase.
   - Remove punctuation.
   - Remove stop words using NLTK.

   ```python
   df['cleaned_text'] = df['review'].str.lower().str.replace('[^\\w\\s]', '', regex=True)
   df['cleaned_text'] = df['cleaned_text'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))
   ```

4. **Generate Word Cloud**: Create a word cloud visualization from the cleaned text.

   ```python
   text = ' '.join(df['cleaned_text'])
   wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
   plt.imshow(wordcloud, interpolation='bilinear')
   plt.axis('off')
   plt.show()
   ```

5. **Determine Optimal Number of Topics**: Implement a function to find the optimal number of topics based on coherence scores.

   ```python
   def determine_optimal_topics(corpus, dictionary, texts, max_topics=10):
       coherence_scores = []
       for num_topics in range(2, max_topics + 1):
           model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, random_state=100)
           coherence_model = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
           coherence_scores.append(coherence_model.get_coherence())
       optimal_topics = coherence_scores.index(max(coherence_scores)) + 2
       return optimal_topics
   ```

6. **Topic Modeling and Visualization**: Train the LDA model and generate visualizations using pyLDAvis.

   ```python
   lda_model, topic_assignments, topic_probabilities_list, vis, df = topic_modeling(corpus, dictionary, num_topics, df)
   pyLDAvis.save_html(vis, 'lda_topic_visualization.html')
   ```

7. **Save Results**: Export the results, including review assignments to topics and topic-word mappings, to CSV and Excel files.

   ```python
   df.to_csv('reviews_with_topics.csv', index=False)
   topic_frequency_table.to_excel('topic_frequency_distribution.xlsx', index=False)
   ```

## Conclusion

This project demonstrates how to perform topic modeling on reviews using LDA. The results can help in understanding the main themes present in the reviews, providing insights for further analysis.

## License

This project is licensed under the MIT License.
