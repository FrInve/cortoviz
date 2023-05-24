from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from bertopic import BERTopic
from scipy.stats import kruskal

import src.topics
from src.visualization.timeline import *
from src.visualization.wc import create_wordcloud

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

# @st.cache_data
# def load_topic_data():
#     df_norm = pd.read_csv('./data/processed/topics_freq_pivoted.csv')
#     df_norm['Date'] = pd.to_datetime(df_norm['Date'])
#     df_norm = df_norm.set_index('Date')
#     return df_norm

# @st.cache_data
# def load_topic_abs_data():
#     df_abs_freq = pd.read_parquet('./data/processed/topics.parquet')
#     df_abs_freq['Date'] = df_abs_freq.Timestamp.dt.strftime('%Y/%m/%d')
#     df_abs_freq['Date'] = pd.to_datetime(df_abs_freq['Date'])
#     df_abs_freq['Topic'] = df_abs_freq.Topic.astype(str)
#     df_abs_freq = df_abs_freq.loc[df_abs_freq.index.repeat(df_abs_freq.Frequency)]
#     df_abs_freq = df_abs_freq[['Date','Topic']]
#     df_abs_freq = df_abs_freq.set_index('Date')
#     return df_abs_freq

# @st.cache_data
# def load_topic_abs_data_aggr():
#     df_abs_freq = pd.read_parquet('./data/processed/topics.parquet')
#     df_abs_freq['Date'] = df_abs_freq.Timestamp.dt.strftime('%Y/%m/%d')
#     df_abs_freq['Date'] = pd.to_datetime(df_abs_freq['Date'])
#     df_abs_freq['Topic'] = df_abs_freq.Topic.astype(str)
#     df_abs_freq = df_abs_freq[['Topic','Frequency','Date']].pivot(index='Date',columns='Topic',values='Frequency').copy(deep=True)
#     return df_abs_freq

@st.cache_resource
def load_topic_model():
    topic_model = BERTopic.load('./models/BERTopic_full_2023-04-18',embedding_model="pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb")
    return topic_model

@st.cache_resource
def create_wordcloud_static(topic):
    return create_wordcloud(topic_model, topic).to_image()

@st.cache_data
def load_topic_data_raw():
    paths = [
        './data/processed/cortoviz_data/131_dyn_raw.csv',
        './data/processed/cortoviz_data/66_dyn_raw.csv',
        './data/processed/cortoviz_data/44_dyn_raw.csv',
        './data/processed/cortoviz_data/33_dyn_raw.csv'
    ]
    dfs = [pd.read_csv(path) for path in paths]
    res = []
    for df in dfs:
        df['Date'] = pd.to_datetime(df['Date'])
        df['Topic'] = df.Topic.astype(str)
        df = df.loc[df.index.repeat(df.Frequency)]
        df = df[['Date','Topic']]
        df = df.set_index('Date')
        res.append(df)
    return res



@st.cache_data
def load_topic_data_pivot():
    paths = [
        './data/processed/cortoviz_data/131_dyn_pivot.parquet',
        './data/processed/cortoviz_data/66_dyn_pivot.parquet',
        './data/processed/cortoviz_data/44_dyn_pivot.parquet',
        './data/processed/cortoviz_data/33_dyn_pivot.parquet'
    ]
    return [pd.read_parquet(path) for path in paths]

@st.cache_data
def load_topic_data_norm():
    paths = [
        './data/processed/cortoviz_data/131_dyn_pivot_norm.parquet',
        './data/processed/cortoviz_data/66_dyn_pivot_norm.parquet',
        './data/processed/cortoviz_data/44_dyn_pivot_norm.parquet',
        './data/processed/cortoviz_data/33_dyn_pivot_norm.parquet'
    ]
    return [pd.read_parquet(path) for path in paths]


dfs_topic = {
    'raw':load_topic_data_raw(),
    'pivot':load_topic_data_pivot(),
    'norm':load_topic_data_norm(),
    #'bins':[131,66,44,33] # This is the number of bins
    'bins':[10,18,27,33] # This is the binwidth, or the number of days per bin
    # TODO: Fix number of bins
}

df_covid_cases = load_cases_data()
covid_timeline = load_timeline_data()
#df_norm = load_topic_data()
df_norm = dfs_topic['norm'][1]
#df_abs_freq = load_topic_abs_data()
df_abs_freq = dfs_topic['raw'][1]
#df_abs_freq_aggr = load_topic_abs_data_aggr()
df_abs_freq_aggr = dfs_topic['pivot'][1]
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

    st.subheader("Inspect Topics' WordClouds:")

    col1, col2, col3 = st.columns(3)
    columns = [col1,col2,col3]

    if 'topics' not in st.session_state:
        st.session_state['topics'] = src.topics.Topics(query, similar_topic)
    else:
        st.session_state.topics.update(query, similar_topic)
    

    for i, topic in enumerate(similar_topic):
        with columns[i%3]:
            st.write(f"No. {i+1} - Similarity: {similarity[i]:.2f}")# - {topic_model.get_topic(topic)[0][0]}")
            st.image(create_wordcloud_static(topic))
            chosen_val = st.checkbox(f"Topic ID: {str(topic)}",value=True)
            st.session_state.topics.select_topic(topic, chosen_val)
            
            if i < 3:
                st.divider()
    
    if 'resolution' not in st.session_state:
        st.session_state['resolution'] = 2

def set_resolution(resolution):
    st.session_state['resolution'] = resolution

with wcol2:
    st.divider()
    resolution = st.radio(label="Resolution (No. of weeks):",options=[1,2,3,4],index=1,help="Each point represents the number of abstracts regarding a topic that were published in the selected number of weeks",horizontal=True)
    st.session_state.resolution = resolution
    # Select the dataset with the appropriate resolution at runtime
    df_norm = dfs_topic['norm'][resolution-1]
    df_abs_freq = dfs_topic['raw'][resolution-1]
    df_abs_freq_aggr = dfs_topic['pivot'][resolution-1]
    bins = dfs_topic['bins'][resolution-1]

    df_tmp = df_norm[df_norm.index < '2022-06-06'][st.session_state.topics.get_selected_topics()].copy()
    df_abs_tmp = df_abs_freq[(df_abs_freq.Topic.isin(st.session_state.topics.get_selected_topics())) & (df_abs_freq.index < '2022-06-06')].copy()
    # Set same colour palette among different plots
    palette_colors = sns.color_palette('tab10')
    palette_dict = {topic:color for topic,color in zip((str(x) for x in similar_topic),palette_colors)}
    fig, (ax1, ax1_bis) = plt.subplots(2,height_ratios=[0.87,0.13],figsize=(11.7,8.27),sharex=True)
    #sns.set_theme()
    sns.lineplot(data=df_tmp, dashes=False, palette=palette_dict, ax=ax1).set(title=query, ylabel="Relative Frequency (%)")
    ax1.yaxis.set_major_formatter(major_formatter_perc)
    sns.histplot(data=df_abs_tmp, multiple="stack", x="Date",hue="Topic", palette=palette_dict, binwidth=bins, legend=False) # Default binwidth 18
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
        st.write(f"Number of bins: {len(stat_first_samples)} - Number of considered abstracts: {int(sum(stat_first_samples_count)[0])}")

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
        st.write(f"Number of bins: {len(stat_second_samples)} - Number of considered abstracts: {int(sum(stat_second_samples_count)[0])}")

        st.divider()
        r_col1, r_col2 = st.columns(2)
        stat_kruskal = kruskal(stat_first_samples, stat_second_samples)
        with r_col1:
            if stat_kruskal.pvalue > 0.05:
                st.markdown(f"The observations of topic {stat_selected_topic} in an interval from {stat_first_date_ranges[0].strftime('%Y-%m-%d')} to {stat_first_date_ranges[1].strftime('%Y-%m-%d')} are **NOT** statistically different from the observations of the interval from {stat_second_date_ranges[0].strftime('%Y-%m-%d')} to {stat_second_date_ranges[1].strftime('%Y-%m-%d')}.  \n  (p-threshold: 5%)")
            else:
                st.markdown(f"The observations of topic {stat_selected_topic} in an interval from {stat_first_date_ranges[0].strftime('%Y-%m-%d')} to {stat_first_date_ranges[1].strftime('%Y-%m-%d')} **ARE** statistically different from the observations of the interval from {stat_second_date_ranges[0].strftime('%Y-%m-%d')} to {stat_second_date_ranges[1].strftime('%Y-%m-%d')}.  \n  (p-threshold: 5%)")
        with r_col2:
            st.write(f"p-value: {stat_kruskal.pvalue[0]:.5f}")
            st.write(f"H statistic: {stat_kruskal.statistic[0]:.5f}")
            st.markdown("[Kruskal-Wallis test](https://en.wikipedia.org/wiki/Kruskal%E2%80%93Wallis_one-way_analysis_of_variance)")

st.markdown("Copyright (C) 2023 Francesco Invernici, All Rights Reserved")