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

# get stopwords to remove from wordcloud and clean comments
nltk.download('stopwords')
stopwords_list = stopwords.words('english')

# dictionary to convert between col names to more descriptive
# names
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

# order of ticks (sequential) for employee size
order_of_ticks = ['1-5', '6-25', '26-100', '100-500', '500-1000', 'More than 1000']

# function to build a word cloud from a given dataframe based on comments col
# applies cleaning: lowercase, stopword removal, regex sorting
def get_wordcloud_info(df):
    word_counts = Counter()
    for message in df['comments'].dropna():
        message = re.sub(r'[^a-zA-Z\s]', '', str(message))
        if isinstance(message, str):
            words = [word.lower() for word in str(message).split() if word.lower() not in stopwords_list]
            word_counts.update(words)
        
    return word_counts

# styling of app
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

# read dataframe
df = pd.read_csv('survey.csv')

# def get_writeup():
#     return """
# <div> insert writeup here </div>
# """

# make main area
with st.container():
    
    # logo icon
    image = Image.open('head.png')
    st.image(image)

    # our project title and subtitle
    st.header('Mental Health Tech Survey')
    st.write("Crack the Code of your own well being with our detailed analysis.")

    st.write('The “Mental Health in Tech Survey” dataset contains the results of a 2014 survey that measures attitudes towards mental health and the frequency of mental health disorders in the workplace, with an emphasis on tech-related jobs. As the tech industry continues to evolve at a rapid pace, the mental well-being of individuals working within it becomes an increasingly significant aspect to consider when looking at new jobs. In the following data analysis, we aim to not only shed light on general feelings towards mental health in the workplace but also specifically address the unique challenges faced by those in the technology sector.')
    
    # bias section
    st.subheader("Understanding the Biases of the Dataset")
    st.write('However, every dataset has limitations based on a limited sample size. Acknowledging the limitations inherent in every dataset is fundamental to ensuring the reliability and applicability of research findings. In this first section, we hope you can better understand who exactly responded to the mental health survey and how that might affect the final results and our continued analysis. If the worker demographics are very different from your own background, you should carefully consider the extent to which the results can be realistically extrapolated to your personal context.')
    
    st.write('This dataset in particular has higher rates of individuals that are male, tech, non-remote, US-based, aged 25-35, and not-self-employed. This suggests marginalized individuals may not have the same experiences as outlined in this dataset. It is also likely that the survey was conducted by data scientists based on the US themselves, which may have heavily influenced the results.')
    
    # sort countries to only look at countries with at least 7 people surveyed
    country_options = df.Country.unique()
    country_options = [country for country in country_options if (df['Country'] == country).sum() > 7]

    # country select
    options = st.multiselect(
        'Select locations to filter plots for worker demographics and job statistics.', sorted(country_options), default=["Canada", "Netherlands"])
    
    # if no selected country, display this:
    if not options:
        st.header("You must select at least one Country.")
        exit()

    # demographics section
    st.subheader('Worker Demographics')
    
    # make new df based on country/countries selection
    filtered_df = df[df['Country'].isin(options)]
    
    # count how many people in the selected county/countries selection
    st.write('There are ' + str(len(filtered_df)) + ' surveyed workers in the selected region(s).')
    
    # side by side section (I'm using this as a janky way to mimic css float property)
    col7, col8 = st.columns(2)
    
    with col7:
        st.write('')
    with col8:
        kde = st.checkbox('Show KDE (Kernel density estimate) line')
        # option to see KDE line on plot

    # build plot for gender distribution and age distribution
    fig, (ax_gender, ax_age) = plt.subplots(1, 2, figsize=(12, 5))

    # pie chart for gender distribution
    gender_mapping = {'m': 'male', 'male': 'male', 'f': 'female', 'female': 'female'}
    filtered_df['Gender'] = filtered_df['Gender'].str.lower().map(gender_mapping).fillna('other')
    gender_counts = filtered_df['Gender'].value_counts()

    ax_gender.set_title('Gender Distribution')
    ax_gender.pie(gender_counts, labels=gender_counts.index, autopct='%1.f%%', startangle=90, colors=['skyblue', 'pink', 'gray'])
    ax_gender.axis('equal')

    # histogram for age distribution
    sns.histplot(filtered_df['Age'], bins=range(0, 101, 5), kde=kde, color='lime', ax=ax_age) # kde based on user input earlier
    ax_age.set_title('Age Distribution')
    ax_age.set_xlabel('Respondent Age')
    ax_age.set_ylabel('Respondent Count')
    ax_age.set_xlim(0, 100)
    
    ax_age.spines['top'].set_visible(False)
    ax_age.spines['right'].set_visible(False)
    
    bin_edges = range(0, 101, 10)
    ax_age.set_xticks(bin_edges)

    # display the distribution figures
    st.pyplot(fig)
    
    # summarize figure takeaway
    st.markdown('`Surveyed participants were skewed towards a more male population with young and middle aged adults.`')
    
    st.write('The emphasis on tech jobs is also quite interesting, and may imply that the data scientists conducting this research believed tech workers had a different mental health outlook than those working in non-tech jobs.')
    
    st.write('Some specific challenges unique to working in tech could include the constant pressure to stay abreast of rapidly evolving technologies. The fast-paced nature of the tech industry, characterized by frequent updates, new programming languages, and evolving frameworks, can create an environment where professionals feel the need to constantly upskill to remain competitive. This perpetual learning curve, while exciting for some, may also contribute to feelings of imposter syndrome, job insecurity, or burnout.')
    
    st.write('Moreover, the inherent demand for innovation and problem-solving in tech roles can lead to high levels of job-related stress. Tight deadlines, challenging projects, and the expectation of continuous innovation may result in heightened stress levels among tech workers. The nature of the work often involves troubleshooting complex issues, and the responsibility for maintaining and securing digital systems can add an additional layer of stress.')
    
    st.write('Even the process of securing a job within tech can be demanding, with multi-step interviews: take an 8-step online assessment with coding challenges, a couple of technical interviews, a behavioral interview, only to be denied from the job you applied for!')
    
    # job statistics section
    st.subheader('Job Statistics')
    
    st.write('While this depends on the countries you sort by, in general, it seems that the majority of workers surveyed were in the tech sector. The majority of workers also were working in non remote work. However, there seems to be a very interesting trend within the dataset, with regards to these two characteristics: Having a tech job appears to be correlated with working from home.')
    
    st.write('This correlation suggests that individuals employed in technology-related roles often have the flexibility or technological infrastructure that allows them to perform their job duties remotely. This trend could be influenced by various factors, such as the nature of tech work that may rely heavily on digital tools and connectivity, the prevalence of remote-friendly policies within the tech industry, or the increasing integration of virtual collaboration platforms.')
    
    st.write('Below, the outer circle shows what portion of our dataset\'s respondents work remotely, and what portion don\'t work remotely. While the inner circle shows the tech vs non-tech break down of the two groups. Clearly, we can see an uneven distribution between remote and non remote work. Within each group, the distribution of tech and non tech jobs is also imbalanced.')

    
    # add more filters to country-filtered data, based on tech company and remote work columns
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

    # display the donut plot
    st.pyplot(fig)
    
    # summarize figure takeaway
    st.markdown('`The dataset has a majority of non-remote workers and tech job workers. Remote workers are more likely to have a tech job.`')

    st.write("Notably, transitioning to remote work can also bring about a profound impact on mental health, presenting a nuanced landscape of both potential benefits and challenges. On one hand, the flexibility and autonomy associated with working from home can contribute positively to mental well-being. The elimination of commuting stress, the ability to create a personalized and comfortable work environment, and the increased control over one's schedule are factors that may enhance job satisfaction and reduce overall stress levels.")
    
    st.write("This isn’t to say that remote work is without its downsides. Isolation and blurred boundaries between work and personal life in a remote setup can also pose challenges to mental health. The absence of face-to-face interactions with colleagues, reduced social engagement, and potential feelings of professional isolation may contribute to a sense of loneliness or disconnection. Additionally, the constant availability created by remote work can lead to difficulties in setting clear boundaries between work and personal time, potentially resulting in burnout.")
    


    # sorting filters section
    st.subheader('Self-employment and employee-base size')
    
    st.markdown('The relationship between having a tech job, working remote, or being self-employeed is very interesting! For example, you are most likely to be self-employed if you work in tech and are remote.')

    st.write("Sort employee counts and self-employment plots by tech and remote filters:")
   
    # divide container into four columns to put each checkbox
    col0, col1, col2, col3 = st.columns(4)

    # put a checkbox in each column
    with col0:
        tech = st.checkbox('Tech jobs', value=True)
    with col1:
        no_tech = st.checkbox('Non-tech jobs')
    with col2:
        remote = st.checkbox('Remote', value=True)
    with col3:
        no_remote = st.checkbox('Non-remote')
    
    # make a copy of the filtered_df
    extra_filtered_df = filtered_df.copy()
    
    # apply sorting  based on user's checkbox input
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

        # display the figure
        st.pyplot(fig)
    
    else:
        # if no filter selected
        st.text('Please select at least one filter to generate a plot.')
        
    st.write("The entrepreneurial spirit that often characterizes tech professionals might drive some to pursue this self-employment path, via freelancing, consulting, or launching their own startups. Self-employed individuals in the tech space often have the autonomy to define their work arrangements, including the option to work remotely.")

    st.write("However, the autonomy and freedom associated with self-employment in the tech industry also bring forth a set of unique mental health challenges. The flexibility to define one's work arrangements and set individual schedules, while liberating, can create a sense of isolation. Self-employed tech professionals often work independently, lacking the daily interactions and camaraderie found in traditional office environments. The absence of a structured support system can contribute to feelings of loneliness and professional isolation, impacting mental well-being.")
   
    # mental health analysis section
    st.subheader('Mental Health Analysis')
    
    st.write("As you have read about so far, an individual’s mental health is influenced by a myriad of factors. Biological factors, such as genetics and neurochemistry, play a pivotal role, shaping an individual's susceptibility to mental health conditions. Environmental factors, including early life experiences, trauma, and socio-economic circumstances, contribute significantly to mental health outcomes. Additionally, individual lifestyle choices, such as diet, exercise, and sleep patterns, exert a profound impact. Social support systems, relationships, and the broader cultural context also influence mental health, highlighting the interconnected nature of human experiences. The complexity of mental health underscores the need for a holistic approach that addresses biological, psychological, social, and environmental dimensions, emphasizing the importance of tailored interventions and destigmatizing conversations surrounding mental well-being.")

    st.write("While we cannot easily address factors that are biological, factors that are environmental or social are greatly tied to your workplace. Many of these factors not only influence your mental health, but also influence each other. For example, having a larger employee size is positively correlated with the likelihood of having benefits offered at your workplace.")
    
    st.write("In the plot below, we take a look at such factors within the workplace. Darker blue hues correspond to a stronger negative linear relationship. Darker red hues correspond to a stronger positive linear relationship.")

    st.write("Looking at the bottom row of the matrix, we can see how each factor is related to whether someone seeks mental health treatment. We can look at other rows to see how different factors are influenced by other factors.")

    # load the figure generated in mess.ipynb (correlation matrix)
    image = Image.open('output.png')
    st.image(image)
    
    st.subheader('Mental health treatment by workplace environment')
    
    st.write("Certain workplace traits seem to encourage workers to seek treatment, such as knowing if they have anonymity when discussing their mental health or if they have care options at the workplace.")
    
    st.write("On-site care options can range from wellness programs and counseling services to mental health workshops and seminars. By bringing mental health resources into the workplace, employers signal a commitment to prioritizing the well-being of their employees. This not only makes accessing mental health support more convenient but also helps normalize discussions around mental health within the organizational culture. Employees are more likely to seek treatment and engage in preventive measures when these services are easily accessible, reducing potential barriers that might otherwise discourage them from seeking help outside the workplace.")
    
    st.write("In addition to anonymity and on-site care options, fostering a culture of open communication and destigmatizing mental health challenges is vital. Training programs that educate employees and leadership about mental health, resilience, and stress management can contribute to a shared understanding of these issues. This awareness helps create an environment where individuals feel comfortable discussing their mental health concerns without fear of judgment. Encouraging managers and colleagues to actively listen, express empathy, and provide support when needed can contribute significantly to a more compassionate workplace atmosphere.")

    
    # select filters for workplace environment
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

    # display the plot
    st.pyplot(fig)
    
    # inference section
    st.subheader('Mental Health Inference')
    
    st.write("Our cutting-edge predictive model now offers the capacity to estimate the likelihood of individuals seeking mental health treatment with an impressive accuracy rate exceeding 70%. Leveraging advanced data analytics and machine learning algorithms, these models analyze a myriad of factors such as individual behavioral patterns, historical health data, and contextual information. By examining this comprehensive dataset, these predictive tools can identify subtle patterns and correlations that might elude human observation, providing a valuable tool for early intervention and personalized mental health care. The promise of predicting the inclination towards seeking mental health treatment underscores the potential for a more proactive and targeted approach in addressing mental health challenges, ultimately contributing to improved overall well-being and more effective healthcare strategies. Try it out yourself and see if, based on your personal information and workplace environment, if you're likely to seek treatment!")
    
    st.write("Predict whether you will seek mental health treatment, with over 70\% accuracy.")
    
    # user input section for making predictions
    
    # dividing the background information section into three columns    
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
    
    pred = st.button('Predict')  # prediction button
    
    if pred:
        # the following dictionaries are the same mappings that were used to encode 
        # the training data in mess.ipynb
        
        # dictionary to convert countries to numbers
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
        
        # dictionary to convert genders to numbers
        gender_dict = {'Female': 0, 'Male': 1, 'Other': 2}
        
        # dictionary to convert employee size to numbers
        empsize_dict = {'1-5': 1, '6-25': 2, '26-100': 3, '100-500': 4, '500-1000':5, 'More than 1000':6}
        
        # build array based on all user input and convert with dictionaries if necessary
        input_array = [country_dict[Country], gender_dict[Gender], Age, anon, benefits, empsize_dict[employee_size], family_history, 
                       care_options, wellness_prog, seek, conseq, remote_work, self_employed, tech_company]
                
        input_array = pd.to_numeric(input_array, errors='coerce')
        
        # loading model
        with open("model.pkl", 'rb') as file:
            clf = pickle.load(file)
        
        # make prediction
        model_pred = clf.predict([input_array])[0]
        # st.write(model_pred)
        
        # print out result based on prediction
        if model_pred:  # predicted 1
            st.write('You are likely to seek mental health treatment.')
        else:  # predicted 0
            st.write('You are unlikely to seek mental health treatment.') 
    
    # comments analysis section
    st.subheader('Worker comments')
    
    st.write('Many workers also have comments about mental health in the workplace too.')
    
    st.write("The survey's inclusion of an open-ended comments section provided a valuable opportunity for respondents to express their individual perspectives, shedding light on the nuanced aspects of the relationship between job benefits and mental health. What emerged from the analysis was a surprising consistency in the verbiage and tone across groups, regardless of whether individuals received job benefits or not. Initial expectations of encountering more negative emotions and distinct language among those not offered benefits were not fully met.")
    
    # select a dropdown option to sort the wordcloud
    option = st.selectbox(
    'Filter wordcloud based on available mental health benefits',
    ('Benefits offered', 'No benefits', 'No filter'))
    
    if option == 'No filter':
        # if no filter, look at the whole dataset (except rows with no comments)
        wcloud_sorted_df = df.dropna(subset=['comments'])
    else:
        # apply filters and look at appropriate data
        wcloud_sorted_df = df[df['benefits'] == 'Yes'] if option == 'Benefits offered' else df[df['benefits'] == 'No']
        wcloud_sorted_df = wcloud_sorted_df.dropna(subset=['comments'])
    
    wcloud_sorted_df = wcloud_sorted_df.reset_index(drop=True)
    word_counts = get_wordcloud_info(wcloud_sorted_df)  # get wordcloud
    num_records = len(wcloud_sorted_df)  # calculate how many people made a comment in filtered data
    
    st.write("Analyzed " + str(num_records) + " comments.")
    
    wordcloud = WordCloud(background_color='white', width=1000, height=500).generate_from_frequencies(word_counts).to_image()
    
    st.image(wordcloud)  # display wordcloud
    
    # summarize figure takeaway
    st.markdown('`Verbiage seems consistent for employees regardless of benefits.`')
    
    st.write("The unexpected uniformity in language and tone suggests a shared sentiment among respondents, emphasizing a commonality in the challenges individuals face regarding mental health in the workplace. This finding underscores a broader concern transcending the specific context of job benefits, indicating that the treatment of mental health is a pervasive issue affecting professionals across various employment situations.")
    
    st.write("A few workers’ comments also expressed their approval of the survey itself, and how they were thankful people were looking into the issue. People being able to express themselves seemed important. These positive sentiments underscore the importance of giving voice to employees' perspectives and the impact it can have on fostering a culture of openness and understanding.")
    
    st.write("You can view a sample comment based from the filtered data (whether you chose benefits or not in the wordcloud dropdown) by clicking the button below.")
    
    # button to see a sample comment in the filtered data
    view_comment = st.button('View a sample comment')
    
    # if button is pressed
    if view_comment:
        random_index = np.random.randint(0, len(wcloud_sorted_df))
        random_comment = wcloud_sorted_df.loc[random_index, 'comments']
        st.write(str(random_comment))

    st.markdown("**Hopefully, you now have a better understanding of the dynamics of mental health in the workplace. The next time you seek out a job, consider if it will bode well for your own well-being. You’re also encouraged to use our machine-learning model to test your likelihood of seeking mental health treatment when applying for your next job! Be sure to always prioritize your health over professional commitments. You got this!**")

    st.markdown("Kaggle. (2016). *Mental Health in Tech Survey.* Open Sourcing Mental Illness, LTD. https://www.kaggle.com/datasets/osmi/mental-health-in-tech-survey.")

    # unused
    # st.subheader("Write Up")
    # st.write(get_writeup(), unsafe_allow_html=True)
