from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import pickle
import streamlit as st
from PIL import Image
import seaborn as sns
from collections import Counter
from wordcloud import WordCloud
import re
from nltk.corpus import stopwords

nltk.download('stopwords')
stopwords_list = stopwords.words('english')

def get_wordcloud_info(df):

    word_counts = Counter()
    for message in df['comments'].dropna():
        message = re.sub(r'[^a-zA-Z\s]', '', str(message))
        if isinstance(message, str):
            words = [word.lower() for word in str(message).split() if word.lower() not in stopwords_list]
            word_counts.update(words)
        
    return word_counts

st.markdown(
     f"""
     <style>
      
    .stApp {{
            background: url("https://cdn.dribbble.com/users/4049548/screenshots/15453291/brain_pattern-01.png");
            background-size: cover;
            filter: hue-rotate(180deg);
        }}
        
    [data-testid="stHeader"]{{
        background-color: black
    }}
        
    *{{
        color: white;
        text-align: center
    }}

    .css-1n76uvr.e1tzin5v0 {{
        background-color: rgba(100, 0, 0, 0.9);
        border-radius: 25px;
        padding: 30px
    }}
    
    [data-baseweb="popover"]>div>div>ul {{
        background-color: #17345F
    }}
    
    [data-baseweb="popover"]>div>div>ul>div>div>li>div>div>div:hover {{
        color: black
    }}
    
    .css-1v0mbdj.etr89bj1{{
        display: block;
        margin: 0 auto;.
        max-width: 100px;
        border: red;
        filter: hue-rotate(-180deg)
    }}
    
    [data-testid="stImage"]{{
        margin-left: 0;
        padding: 0;
        filter: hue-rotate(-180deg)
    }}
    
    .st-f2.st-bc.st-d8.st-f3.st-d2.st-be.st-cq.st-cp.st-f4{{
        background-color: red !important
    }}
    
    button >.css-1offfwp.e16nr0p34 p{{
        color: black
    }}
    
     </style>
     """,
     unsafe_allow_html=True
 )


# loading
# with open("model.pkl", 'rb') as file:
#     clf = pickle.load(file)

df = pd.read_csv('survey.csv')
    
with st.container():
    
    image = Image.open('head.png')
    st.image(image)

    st.header('Mental Health Tech Survey')
    st.write("Insert a byline")
    
    country_options = df.Country.unique()
    options = st.multiselect(
        'Select locations to filter plots for worker demographics and job statistics.', country_options, default=["Canada", "Netherlands"])
    
    st.subheader('Worker Demographics')
    
    filtered_df = df[df['Country'].isin(options)]
    
    st.write('There are ' + str(len(filtered_df)) + ' surveyed workers in the selected region(s).')

    fig, (ax_gender, ax_age) = plt.subplots(1, 2, figsize=(12, 5))

    # pie chart for gender distribution
    gender_mapping = {'m': 'male', 'male': 'male', 'f': 'female', 'female': 'female'}
    filtered_df['Gender'] = filtered_df['Gender'].str.lower().map(gender_mapping).fillna('other')
    gender_counts = filtered_df['Gender'].value_counts()

    ax_gender.set_title('Gender Distribution')
    ax_gender.pie(gender_counts, labels=gender_counts.index, autopct='%1.f%%', startangle=90, colors=['skyblue', 'pink', 'gray'])
    ax_gender.axis('equal')

    # histogram for age distribution
    sns.histplot(filtered_df['Age'], bins=10, kde=False, color='skyblue', ax=ax_age)
    ax_age.set_title('Age Distribution')
    ax_age.set_xlabel('Age')
    ax_age.set_ylabel('Count')
    ax_age.set_xlim(0, 100)
    
    ax_age.spines['top'].set_visible(False)
    ax_age.spines['right'].set_visible(False)
    
    bin_edges = range(0, 101, 10)
    ax_age.set_xticks(bin_edges)

    st.pyplot(fig)
    
    st.markdown('`Surveyed participants were skewed towards a more male population with young and middle aged adults.`')
        
    st.subheader('Job Statistics')
    
    counts = filtered_df.groupby(['tech_company', 'remote_work']).size().unstack().fillna(0).stack()

    # nested donut plot
    fig, ax = plt.subplots()

    size = 0.3
    vals = np.array([counts['No'].values, counts['Yes'].values])

    cmap = plt.colormaps["tab20c"]
    outer_colors = cmap([1, 12])
    inner_colors = cmap([20, 16, 20, 16])

    # outer circle for remote work
    outer_wedges, outer_texts = ax.pie(vals.sum(axis=1), radius=1, colors=outer_colors,
                                                        wedgeprops=dict(width=size, edgecolor='w'))

    # inner circle for tech company
    inner_wedges, inner_texts = ax.pie(vals.flatten(), radius=1-size, colors=inner_colors,
                                                        wedgeprops=dict(width=size, edgecolor='w'))

    ax.set(aspect="equal", title='What is the relationship between Tech Jobs and Remote Work?')

    ax.legend(outer_wedges + inner_wedges, ['Remote Work', 'Not Remote Work', 'Tech Job', 'Non Tech Job'],
            loc='center left', bbox_to_anchor=(1, 0.5))

    st.pyplot(fig)
    
    st.markdown('`The dataset has a majority of non-remote workers and tech job workers. Remote workers are more likely to have a tech job.`')

    st.write('Sort employee counts and self-employment plots by tech and remote filters:')
    
    col0, col1, col2, col3 = st.columns(4)

    with col0:
        tech = st.checkbox('Tech jobs', value=True)
    with col1:
        no_tech = st.checkbox('No tech jobs')
    with col2:
        remote = st.checkbox('Remote', value=True)
    with col3:
        no_remote = st.checkbox('No remote')
    
    extra_filtered_df = filtered_df.copy()
    
    if (tech ^ no_tech):
        if tech:
            extra_filtered_df = extra_filtered_df[extra_filtered_df['tech_company'] == 'Yes']

        if no_tech:
            extra_filtered_df = extra_filtered_df[extra_filtered_df['tech_company'] == 'No']
            
    if (remote ^ no_remote):
        if remote:
            extra_filtered_df = extra_filtered_df[extra_filtered_df['remote_work'] == 'Yes']

        if no_remote:
            extra_filtered_df = extra_filtered_df[extra_filtered_df['remote_work'] == 'No']
            
    if (tech or no_tech or remote or no_remote):

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # bar chart for 'no_employees'
        order_of_ticks = ['1-5', '6-25', '26-100', '100-500', '500-1000', 'More than 1000']
        tick_labels = ['1-5', '6-25', '26-100', '100-500', '500-1000', '1000+']

        sns.countplot(x='no_employees', data=extra_filtered_df, ax=ax1, palette='viridis', order=order_of_ticks)
            
        ax1.set_title('Employee Count Distribution')
        ax1.set_xlabel('Number of Employees')
        ax1.set_ylabel('Count')
        ax1.set_xticklabels(tick_labels)
        
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)

        # pie chart for 'self_employed'
        self_employed_counts = extra_filtered_df['self_employed'].value_counts()
        ax2.pie(self_employed_counts, labels=self_employed_counts.index, autopct='%1.1f%%', startangle=90, colors=['orange', 'green'])
        ax2.axis('equal')
        ax2.set_title('Self-Employment Distribution')

        st.pyplot(fig)
    
    else:
        st.text('Please select at least one filter to generate a plot.')
        
    st.subheader('Mental Health Analysis')
    
    image = Image.open('output.png')
    st.image(image)
    
    st.write('Darker blue hues correspond to a stronger negative linear relationship. Darker red hues correspond to a stronger positive linear relationship.')
    
    st.subheader('Mental Health Inference')
    
    st.subheader('Worker comments')
    
    option = st.selectbox(
    'Filter wordcloud based on available mental health benefits',
    ('Benefits offered', 'No benefits', 'No filter'))
    
    if option == 'No filter':
        wcloud_sorted_df = df.dropna(subset=['comments'])
    else:
        wcloud_sorted_df = df[df['benefits'] == 'Yes'] if option == 'Benefits offered' else df[df['benefits'] == 'No']
        wcloud_sorted_df = wcloud_sorted_df.dropna(subset=['comments'])
    wcloud_sorted_df = wcloud_sorted_df.reset_index(drop=True)
    word_counts = get_wordcloud_info(wcloud_sorted_df)
    num_records = len(wcloud_sorted_df)
    # st.write(wcloud_sorted_df.benefits.value_counts())
    
    wordcloud = WordCloud(background_color='white', width=1000, height=500).generate_from_frequencies(word_counts).to_image()
    
    st.image(wordcloud)
    st.write("Analyzed " + str(num_records) + " comments.")
    
    st.markdown('`Verbage seems consistent for employees regardless of benefits.`')
        
    view_comment = st.button('View a sample comment')
    
    if view_comment:
        random_index = np.random.randint(0, len(wcloud_sorted_df))
        random_comment = wcloud_sorted_df.loc[random_index, 'comments']
        st.write(str(random_comment))


    st.subheader("Write Up")
