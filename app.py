from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import pickle
from PIL import Image
import seaborn as sns
from collections import Counter
from wordcloud import WordCloud
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stopwords_list = stopwords.words('english')

select_dict = {
    'have anonymity when discussing mental health': 'anonymity',
    'are offered benefits': 'benefits',
    'have bosses who are more biased against mental health vs physical health': 'mental_vs_physical',
    'have a family history of mental health issues': 'family_history',
    'have care options at work': 'care_options',
    'have a wellness program at work': 'wellness_program',
    'feel that asking for help is encouraged': 'seek_help',
    'have observed negative consequences from reporting mental health problems': 'obs_consequence',
}

order_of_ticks = ['1-5', '6-25', '26-100', '100-500', '500-1000', 'More than 1000']

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
    
    div[data-baseweb="select"] > div {{
        background-color: black;
    }}
    
    button >.css-1offfwp.e16nr0p34 p{{
        color: black
    }}
    
    .e1fb0mya1.css-bubqsq.ex0cdmw0, e1fb0mya1.css-bubqsq.ex0cdmw0{{
        fill: red
    }}
    
     </style>
     """,
     unsafe_allow_html=True
 )

df = pd.read_csv('survey.csv')

def get_writeup():
    return """
<div> insert writeup here </div>
"""

with st.container():
    
    image = Image.open('head.png')
    st.image(image)

    st.header('Mental Health Tech Survey')
    st.write("Insert a byline")
    
    country_options = df.Country.unique()
    country_options = [country for country in country_options if (df['Country'] == country).sum() > 7]

    options = st.multiselect(
        'Select locations to filter plots for worker demographics and job statistics.', sorted(country_options), default=["Canada", "Netherlands"])
    
    if not options:
        st.header("You must select at least one Country.")
        exit()


    st.subheader('Worker Demographics')
    
    filtered_df = df[df['Country'].isin(options)]
    
    st.write('There are ' + str(len(filtered_df)) + ' surveyed workers in the selected region(s).')
    
    col7, col8 = st.columns(2)
    
    with col7:
        st.write('')
    with col8:
        kde = st.checkbox('Show KDE (Kernel density estimate) line')

    
    fig, (ax_gender, ax_age) = plt.subplots(1, 2, figsize=(12, 5))

    # pie chart for gender distribution
    gender_mapping = {'m': 'male', 'male': 'male', 'f': 'female', 'female': 'female'}
    filtered_df['Gender'] = filtered_df['Gender'].str.lower().map(gender_mapping).fillna('other')
    gender_counts = filtered_df['Gender'].value_counts()

    ax_gender.set_title('Gender Distribution')
    ax_gender.pie(gender_counts, labels=gender_counts.index, autopct='%1.f%%', startangle=90, colors=['skyblue', 'pink', 'gray'])
    ax_gender.axis('equal')

    # histogram for age distribution
    sns.histplot(filtered_df['Age'], bins=range(0, 101, 5), kde=kde, color='lime', ax=ax_age)
    ax_age.set_title('Age Distribution')
    ax_age.set_xlabel('Respondent Age')
    ax_age.set_ylabel('Respondent Count')
    ax_age.set_xlim(0, 100)
    
    ax_age.spines['top'].set_visible(False)
    ax_age.spines['right'].set_visible(False)
    
    bin_edges = range(0, 101, 10)
    ax_age.set_xticks(bin_edges)

    st.pyplot(fig)
    
    st.markdown('`Surveyed participants were skewed towards a more male population with young and middle aged adults.`')
        
    st.subheader('Job Statistics')
    st.write('The Outer Circle shows what portion of our dataset\'s respondents work remotely, and what portion don\'t work remotely. While the inner circle shows the tech vs non-tech break down of the two groups.')
    
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


    st.subheader('Sort employee counts and self-employment plots by tech and remote filters:')
    

    col0, col1, col2, col3 = st.columns(4)

    with col0:
        tech = st.checkbox('Tech jobs', value=True)
    with col1:
        no_tech = st.checkbox('Non-tech jobs')
    with col2:
        remote = st.checkbox('Remote', value=True)
    with col3:
        no_remote = st.checkbox('Non-remote')
    
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
        tick_labels = ['1-5', '6-25', '26-100', '100-500', '500-1000', '1000+']

        sns.countplot(x='no_employees', data=extra_filtered_df, ax=ax1, palette='viridis', order=order_of_ticks)
            
        ax1.set_title('Employee Count Distribution')
        ax1.set_xlabel('Number of Employees')
        ax1.set_ylabel('Respondent Count')
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

    st.divider()    

    st.subheader('Mental Health Analysis')
    st.write('Static analysis of the entire dataset showing what factors correlate positively and negatively')
    
    image = Image.open('output.png')
    st.image(image)


    
    st.write('Darker blue hues correspond to a stronger negative linear relationship. Darker red hues correspond to a stronger positive linear relationship.')
    
    st.divider()
    selectbox_filter = st.selectbox('View workers that', ('are offered benefits', 'have care options at work',
                                    'have bosses who are more biased against mental health vs physical health', 
                                    'have a family history of mental health issues', 'have anonymity when discussing mental health',
                                    'have a wellness program at work', 'feel that asking for help is encouraged',
                                    'have observed negative consequences from reporting mental health problems',
                                    ))
    
    select_filter = select_dict[selectbox_filter]
    sorted_labels = df[select_filter].value_counts().index

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    colors = plt.cm.Dark2.colors

    # pie chart
    ax1.pie(df[select_filter].value_counts()[sorted_labels], labels=sorted_labels, autopct='%1.f%%', startangle=90, colors=colors)
    ax1.set_title(str(select_filter).capitalize() + ' Distribution')

    # grouped bar chart
    grouped_data = df.groupby(['treatment', select_filter]).size().unstack()
    grouped_data = grouped_data[sorted_labels]
    grouped_data.plot(kind='bar', stacked=False, ax=ax2, legend=False, color=colors)
    ax2.set_title(str(select_filter).capitalize() + ' vs Treatment   ')
    ax2.set_xlabel('Treatment')
    ax2.set_ylabel('Worker Count')
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(['No treatment', 'Getting treatment'], rotation=0)

    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position("right")
    ax2.spines['top'].set_visible(False)
    ax2.spines['left'].set_visible(False)

    st.pyplot(fig)
    
    st.subheader('Mental Health Inference')
    
    st.write("Predict whether you will seek mental health treatment, with over 70\% accuracy.")
    
    col4, col5, col6 = st.columns(3)
    with col4:
        Gender = st.selectbox('Gender', ('Male', 'Female', 'Other'))
        employee_size = st.selectbox('Company size (num employees)', options = order_of_ticks)

    with col5:
        Age = st.number_input('Age', min_value = 15, max_value= 100, value=20)
        self_employed = st.checkbox('Self Employed?')
        remote_work = st.checkbox('Remote Work?')
        
    with col6:
        Country = st.selectbox('Country', options = country_options)
        tech_company = st.checkbox('Tech Company?')
        benefits = st.checkbox('Benefits?')

    st.write('Click the box if you agree with the statement.')
    
    family_history = st.checkbox('My family has a history of mental illness.')
    care_options = st.checkbox('I know the care options available at my company.')
    wellness_prog = st.checkbox('My employer offers a wellness program.')
    seek = st.checkbox('My employer encourages me to seek outside help for mental health.')
    anon = st.checkbox('My anonymity is protected.')
    conseq = st.checkbox('I have observed negative consequences for coworkers with mental health conditions.')
    
    pred = st.button('Predict')
    
    if pred:
        
        country_dict = {
            'Australia': 0,
            'Canada': 1,
            'France': 2,
            'Germany': 3,
            'India': 4,
            'Ireland': 5,
            'Netherlands': 6,
            'New Zealand': 7,
            'United Kingdom': 8,
            'United States': 9
        }
        
        gender_dict = {'Female': 0, 'Male': 1, 'Other': 2}
        
        empsize_dict = {'1-5': 1, '6-25': 2, '26-100': 3, '100-500': 4, '500-1000':5, 'More than 1000':6}
        
        input_array = [country_dict[Country], gender_dict[Gender], Age, anon, benefits, empsize_dict[employee_size], family_history, 
                       care_options, wellness_prog, seek, conseq, remote_work, self_employed, tech_company]
                
        input_array = pd.to_numeric(input_array, errors='coerce')
        
        # loading model
        with open("model.pkl", 'rb') as file:
            clf = pickle.load(file)
        
        model_pred = clf.predict([input_array])[0]
        # st.write(model_pred)
        
        if model_pred:
            st.write('You are likely to seek mental health treatment.')
        else:
            st.write('You are unlikely to seek mental health treatment.') 
    
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
    st.write(get_writeup(), unsafe_allow_html=True)
