import streamlit as st
import pandas as pd
#from scipy import integrate
from bertopic import BERTopic
import seaborn as sns
import matplotlib.pyplot as plt
from src.visualization.timeline import *
from src.visualization.wc import create_wordcloud
import src.topics
from datetime import datetime,timedelta
from scipy.stats import kruskal

st.set_page_config(layout="wide",
                   page_title="CORToViz",
                   page_icon="📈",
                   menu_items={
                       'About':"Designed and developed by Francesco Invernici, prof. Anna Bernasconi, and prof. Stefano Ceri @DEIB, Politecnico di Milano, Italy."
                   })

@st.cache_data
def load_cases_data():
    df_covid_cases = get_covid_data(end_date='2022-06-15')
    return df_covid_cases

@st.cache_data
def load_timeline_data():
    covid_timeline = pd.read_csv('./data/raw/macmillan_covid.csv')
    covid_timeline['Date'] = pd.to_datetime(covid_timeline['Date'])
    covid_timeline.set_index("Date",inplace=True)
    return covid_timeline

@st.cache_data
def load_topic_data():
    df_norm = pd.read_csv('./data/processed/topics_freq_pivoted.csv')
    df_norm['Date'] = pd.to_datetime(df_norm['Date'])
    df_norm = df_norm.set_index('Date')
    return df_norm

@st.cache_data
def load_topic_abs_data():
    df_abs_freq = pd.read_parquet('./data/processed/topics.parquet')
    df_abs_freq['Date'] = df_abs_freq.Timestamp.dt.strftime('%Y/%m/%d')
    df_abs_freq['Date'] = pd.to_datetime(df_abs_freq['Date'])
    df_abs_freq['Topic'] = df_abs_freq.Topic.astype(str)
    df_abs_freq = df_abs_freq.loc[df_abs_freq.index.repeat(df_abs_freq.Frequency)]
    df_abs_freq = df_abs_freq[['Date','Topic']]
    df_abs_freq = df_abs_freq.set_index('Date')
    return df_abs_freq

@st.cache_data
def load_topic_abs_data_aggr():
    df_abs_freq = pd.read_parquet('./data/processed/topics.parquet')
    df_abs_freq['Date'] = df_abs_freq.Timestamp.dt.strftime('%Y/%m/%d')
    df_abs_freq['Date'] = pd.to_datetime(df_abs_freq['Date'])
    df_abs_freq['Topic'] = df_abs_freq.Topic.astype(str)
    df_abs_freq = df_abs_freq[['Topic','Frequency','Date']].pivot(index='Date',columns='Topic',values='Frequency').copy(deep=True)
    return df_abs_freq

@st.cache_resource
def load_topic_model():
    topic_model = BERTopic.load('./models/BERTopic_full_2023-04-18',embedding_model="pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb")
    return topic_model

@st.cache_resource
def create_wordcloud_static(topic):
    return create_wordcloud(topic_model, topic).to_image()


df_covid_cases = load_cases_data()
covid_timeline = load_timeline_data()
df_norm = load_topic_data()
df_abs_freq = load_topic_abs_data()
df_abs_freq_aggr = load_topic_abs_data_aggr()
topic_model = load_topic_model()

st.markdown(
    """
        <style>
            .appview-container .main .block-container {{
                padding-top: {padding_top}rem;
                padding-bottom: {padding_bottom}rem;
                }}

        </style>""".format(
        padding_top=0, padding_bottom=1
    ),
    unsafe_allow_html=True,
)
st.title("CORToViz")
st.header("The CORD-19 Topics Visualizer")

wcol1, wcol2 = st.columns(2)

with wcol1:
    query = st.text_input("Search a topic:", value='variant', max_chars=100)

    similar_topic, similarity = topic_model.find_topics(query, top_n=6)
    #st.text("Topic\tSimilarity")
    #for topic, sim in zip(similar_topic, similarity):
    #    st.text(f"{topic}\t{sim}")

    st.subheader("Inspect Topics' WordClouds:")

    col1, col2, col3 = st.columns(3)
    columns = [col1,col2,col3]

    if 'topics' not in st.session_state:
        st.session_state['topics'] = src.topics.Topics(query, similar_topic)
    else:
        st.session_state.topics.update(query, similar_topic)
    
    #selecting_topics = {str(topic):True for topic in similar_topic} #######################


    for i, topic in enumerate(similar_topic):
        with columns[i%3]:
            st.write(f"No. {i+1} - Similarity: {similarity[i]:.2f}")# - {topic_model.get_topic(topic)[0][0]}")
            st.image(create_wordcloud_static(topic))
            chosen_val = st.checkbox(f"Topic ID: {str(topic)}",value=True)
            #selecting_topics[str(topic)] = chosen_val ################
            st.session_state.topics.select_topic(topic, chosen_val)
            
            if i < 3:
                st.divider()
        

    #old_selecting_topics = selecting_topics.copy() #############################
    #if 'selected_topics' not in st.session_state:
    #    st.session_state['selected_topics'] = [topic[0] for topic in selecting_topics.items() if topic[1]]
        

with wcol2:
    #st.text(f"Currently showing {len(selected_topics)} topics over time")
    st.divider()

    df_tmp = df_norm[df_norm.index < '2022-06-06'][st.session_state.topics.get_selected_topics()].copy()
    df_abs_tmp = df_abs_freq[(df_abs_freq.Topic.isin(st.session_state.topics.get_selected_topics())) & (df_abs_freq.index < '2022-06-06')].copy()
    # Set same colour palette among different plots
    palette_colors = sns.color_palette('tab10')
    palette_dict = {topic:color for topic,color in zip((str(x) for x in similar_topic),palette_colors)}
    fig, (ax1, ax1_bis) = plt.subplots(2,height_ratios=[0.87,0.13],figsize=(11.7,8.27),sharex=True)
    #sns.set_theme()
    #plt.stackplot(df_tmp.index, df_tmp.iloc[:,0] )
    sns.lineplot(data=df_tmp, dashes=False, palette=palette_dict, ax=ax1).set(title=query, ylabel="Relative Frequency (%)")
    ax1.yaxis.set_major_formatter(major_formatter_perc)
    sns.histplot(data=df_abs_tmp, multiple="stack", x="Date",hue="Topic", palette=palette_dict, binwidth=18, legend=False)
    ax2= ax1.twinx()
    ax2.set_ylabel("Worldwide number of COVID-19 active cases (Millions)")
    plot_events(covid_timeline[covid_timeline.Included==1][['Event']].to_dict()['Event'])
    plot_covid_cases(covid_df=df_covid_cases, ax=ax2)
    fig.subplots_adjust(hspace=0)
    st.pyplot(fig)

def show_only_cb(selected_topic):
    st.session_state.topics.toggle_solo(selected_topic)

with st.expander("Test your hypotheses", expanded=True):
    expander_col1, expander_col2 = st.columns(2, gap="large")
    with expander_col1:
        stat_selected_topic = st.selectbox(
            "Select one topic to verify if it changes through time (Kruskal-Wallis Test):",
            similar_topic
        )
        st.checkbox("Show only this topic", value=False, on_change=show_only_cb, kwargs={'selected_topic':stat_selected_topic})
        #if not st.checkbox("Show only this topic", value=False, on_change=show_only_cb, args=(stat_selected_topic, ):
            #selecting_topics = {str(topic):(True if str(topic)==str(stat_selected_topic) else False) for topic in similar_topic} #######
            #st.session_state.topics.set_solo(stat_selected_topic, False)
        #else:
            #st.session_state.topics.set_solo(stat_selected_topic, True)
            #selecting_topics = old_selecting_topics #########

        #st.text(st.session_state.topics.get_solo())

        #st.session_state['selected_topics'] = [topic[0] for topic in selecting_topics.items() if topic[1]] ##########
    with expander_col2:
        stat_first_date_ranges = st.slider(
            "Select the first time interval (YYYY/MM/DD)",
            value=(datetime(2020,3,1,0,0), datetime(2020,9,1,0,0)),
            format="YYYY/MM/DD",
            min_value=datetime(2019,9,1,0,0),
            max_value=datetime(2022,7,31,0,0),
            step=timedelta(days=7)
        )
        stat_first_mask = (df_norm.index >= stat_first_date_ranges[0]) & (df_norm.index <= stat_first_date_ranges[1])
        stat_first_samples = df_norm[[str(stat_selected_topic)]][stat_first_mask].to_numpy(na_value=0)
        stat_first_samples_count = df_abs_freq_aggr[[str(stat_selected_topic)]][stat_first_mask].to_numpy(na_value=0)
        st.write(f"Number of bins: {len(stat_first_samples)} - Number of papers: {int(sum(stat_first_samples_count)[0])}")

        st.divider()
        stat_second_date_ranges = st.slider(
            "Select the second time interval (YYYY/MM/DD)",
            value=(datetime(2021,6,1,0,0), datetime(2021,12,1,0,0)),
            format="YYYY/MM/DD",
            min_value=datetime(2019,9,1,0,0),
            max_value=datetime(2022,7,31,0,0),
            step=timedelta(days=7)
        )
        stat_second_mask = (df_norm.index >= stat_second_date_ranges[0]) & (df_norm.index <= stat_second_date_ranges[1])
        stat_second_samples = df_norm[[str(stat_selected_topic)]][stat_second_mask].to_numpy(na_value=0)
        stat_second_samples_count = df_abs_freq_aggr[[str(stat_selected_topic)]][stat_second_mask].to_numpy(na_value=0)
        st.write(f"Number of bins: {len(stat_second_samples)} - Number of papers: {int(sum(stat_second_samples_count)[0])}")

        r_col1, r_col2 = st.columns(2)
        stat_kruskal = kruskal(stat_first_samples, stat_second_samples)
        with r_col1:
            if stat_kruskal.pvalue > 0.05:
                st.write("There is no statistically significant difference (p-value=5%)")
            else:
                st.markdown("**There is statistically significant difference**  \n  (threshold: 5%)")
        with r_col2:
            st.write(f"p-value: {stat_kruskal.pvalue[0]:.5f}")
            st.write(f"H statistic: {stat_kruskal.statistic[0]:.5f}")


st.markdown("Copyright (C) 2023 Francesco Invernici, All Rights Reserved")